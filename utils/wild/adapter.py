#!/usr/bin/env python3
"""Convert the WILD-raw item-level evaluation dataset into Every Eval Ever records.

WILD-raw (`kensho/WILD-raw`, arXiv:2604.01418) is **item-level** evaluation data:
~7.5M (model, item) rows for 65 models across 27 benchmarks (109,566 items), each
row a single item response — conversation, model answer, target, binary score,
token usage, and scorer output. It is an `evaluation_run` source (Kensho ran the
evals), so `evaluator_relationship = third_party`.

Mapping:
- One aggregate `EvaluationLog` per (model, benchmark): `evaluation_results` = the
  benchmark overall accuracy (`wild.<task>`) plus one per subtask
  (`wild.<task>.<subtask>`), each a `continuous` [0,1] accuracy (mean of the binary
  item scores), with item counts + mean token usage in `additional_details`.
- With `--include-instances`, the per-item rows are written to a
  `<uuid>_samples.jsonl` instance sidecar (instance_level schema) referenced by the
  aggregate's `detailed_evaluation_results` — the faithful use of the *raw* dataset.

Reads parquet directly from HuggingFace (streamed per row group, never the full
7GB) or from local `--parquet` paths.

Run:
    uv run python -m utils.wild.adapter --output-dir /tmp/eee-wild --limit-shards 1
    uv run python -m every_eval_ever validate /tmp/eee-wild
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import timezone
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq

from every_eval_ever.eval_types import (
    DetailedEvaluationResults,
    EvalLibrary,
    EvaluationLog,
    EvaluationResult,
    EvaluatorRelationship,
    Format,
    HashAlgorithm,
    MetricConfig,
    ModelInfo,
    ScoreDetails,
    ScoreType,
    SourceDataHf,
    SourceDataPrivate,
    SourceMetadata,
)
from every_eval_ever.helpers import SCHEMA_VERSION, save_evaluation_log
from every_eval_ever.instance_level_types import (
    AnswerAttributionItem,
    Evaluation,
    Input,
    InstanceLevelEvaluationLog,
    InteractionType,
    Output,
    TokenUsage,
)

HF_REPO_ID = 'kensho/WILD-raw'
HF_REVISION = 'main'
N_SHARDS = 15
DEFAULT_OUTPUT_DIR = 'data/wild'
SOURCE_NAME = 'WILD-raw'
SOURCE_ORGANIZATION = 'Kensho'
HF_DATASET_URL = f'https://huggingface.co/datasets/{HF_REPO_ID}'
PAPER_URL = 'https://arxiv.org/abs/2604.01418'

# WILD task -> the benchmark's source dataset HF repo (each verified to exist on
# HF). `source_data` names the *dataset the eval ran on*, NOT kensho/WILD-raw
# (which holds the results -> that is source_metadata). Tasks with no clean public
# repo use the `other` variant. Canonicalizing the benchmark id (incl. arc -> AI2
# ARC `ai2-reasoning-challenge-arc`, not ARC-AGI) is the eval-card-registry's job;
# see README for the alias/new-canonical follow-ups.
WILD_DATASET_REPO = {
    'arc_easy': 'allenai/ai2_arc', 'arc_challenge': 'allenai/ai2_arc',
    'bbh': 'lukaemon/bbh', 'bigcodebench': 'bigcode/bigcodebench',
    'boolq': 'google/boolq', 'chembench': 'jablonkagroup/ChemBench',
    'commonsense_qa': 'tau/commonsense_qa', 'drop': 'ucinlp/drop',
    'gsm8k': 'openai/gsm8k', 'gsm_symbolic': 'apple/GSM-Symbolic',
    'hellaswag': 'Rowan/hellaswag', 'ifeval': 'google/IFEval',
    'math': 'hendrycks/competition_math', 'medqa': 'bigbio/med_qa',
    'mmlu': 'cais/mmlu', 'mmlu_pro': 'TIGER-Lab/MMLU-Pro', 'musr': 'TAUR-Lab/MuSR',
    'paws': 'google-research-datasets/paws', 'piqa': 'ybisk/piqa',
    'race_h': 'ehovy/race', 'squad': 'rajpurkar/squad',
    'truthfulqa': 'truthfulqa/truthful_qa', 'winogrande': 'allenai/winogrande',
    # provenance resolved from the WILD paper + Inspect Evals loaders:
    'finance_fundamentals': 'kensho/bizbench', 'pre_flight': 'AirsideLabs/pre-flight-06',
    'bbeh': 'BBEH/bbeh',
}
# aime's two subtasks come from different repos (the exact ones Inspect Evals loads).
AIME_REPO_BY_SUBTASK = {'2024': 'Maxwell-Jia/AIME_2024', '2025': 'math-ai/aime25'}

AGG_COLUMNS = ['model', 'task', 'subtask', 'score', 'input_tokens', 'output_tokens']
INSTANCE_COLUMNS = AGG_COLUMNS + ['item_id', 'conversation', 'target', 'answer',
                                  'scores', 'stop_reason']


# --------------------------------------------------------------------------- #
# parquet streaming (HF or local), per row group, column-projected
# --------------------------------------------------------------------------- #

def _shard_handles(parquet: list[str] | None, limit_shards: int | None):
    """Yield (label, opener) for each parquet source. opener() -> file-like."""
    if parquet:
        sources = parquet
    else:
        sources = [
            f'datasets/{HF_REPO_ID}/data-{i:05d}-of-{N_SHARDS:05d}.parquet'
            for i in range(N_SHARDS)
        ]
    if limit_shards is not None:
        sources = sources[:limit_shards]
    for src in sources:
        if parquet:  # local path
            yield src, (lambda s=src: open(s, 'rb'))
        else:        # HuggingFace
            from huggingface_hub import HfFileSystem
            fs = HfFileSystem()
            yield src, (lambda s=src: fs.open(s, revision=HF_REVISION))


def iter_batches(parquet: list[str] | None, columns: list[str],
                 limit_shards: int | None = None) -> Iterator[dict[str, list]]:
    """Yield column-projected row groups as {col: [values]} dicts."""
    for label, opener in _shard_handles(parquet, limit_shards):
        with opener() as fh:
            pf = pq.ParquetFile(fh)
            for rg in range(pf.num_row_groups):
                tbl = pf.read_row_group(rg, columns=columns)
                yield {c: tbl.column(c).to_pylist() for c in columns}


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #

@dataclass
class Agg:
    n: int = 0
    correct: float = 0.0
    in_tok: int = 0
    out_tok: int = 0

    def add(self, score, in_t, out_t):
        self.n += 1
        self.correct += float(score or 0)
        self.in_tok += int(in_t or 0)
        self.out_tok += int(out_t or 0)


def aggregate(parquet, limit_shards, models: set[str] | None):
    """Return {(model, task): {subtask|None: Agg}}. None key = benchmark overall."""
    groups: dict[tuple[str, str], dict[str | None, Agg]] = defaultdict(
        lambda: defaultdict(Agg))
    for batch in iter_batches(parquet, AGG_COLUMNS, limit_shards):
        for model, task, subtask, score, in_t, out_t in zip(
                batch['model'], batch['task'], batch['subtask'],
                batch['score'], batch['input_tokens'], batch['output_tokens']):
            if models and model not in models:
                continue
            g = groups[(model, task)]
            g[None].add(score, in_t, out_t)                       # benchmark overall
            g[subtask if subtask not in (None, '') else '_'].add(score, in_t, out_t)
    return groups


# --------------------------------------------------------------------------- #
# record construction
# --------------------------------------------------------------------------- #

def _source_data(task: str, n: int, subtask: str | None = None):
    """The dataset the eval ran on (not WILD-raw, which holds the results)."""
    repo = WILD_DATASET_REPO.get(task)
    if task == 'aime' and subtask in AIME_REPO_BY_SUBTASK:
        repo = AIME_REPO_BY_SUBTASK[subtask]
    if repo:
        return SourceDataHf(dataset_name=task, source_type='hf_dataset',
                            hf_repo=repo, samples_number=n)
    return SourceDataPrivate(
        dataset_name=task, source_type='other',
        additional_details={'note': 'no single public HF dataset repo for this WILD '
                                    'task/subtask; results are in ' + HF_REPO_ID})


def _result(task: str, subtask: str | None, agg: Agg) -> EvaluationResult:
    name = f'wild.{task}' if subtask is None else f'wild.{task}.{subtask}'
    rid = task if subtask is None else f'{task}::{subtask}'
    accuracy = agg.correct / agg.n if agg.n else 0.0
    # score is binary per item (verified), so accuracy = mean and the analytic
    # standard error of a proportion is sqrt(p(1-p)/n).
    se = math.sqrt(accuracy * (1 - accuracy) / agg.n) if agg.n else 0.0
    level = 'overall' if subtask is None else 'subtask'
    return EvaluationResult(
        evaluation_result_id=rid,
        evaluation_name=name,
        source_data=_source_data(task, agg.n, subtask),
        metric_config=MetricConfig(
            evaluation_description=(
                f'Mean binary item correctness on {name} (WILD-raw).'),
            metric_id=f'{name}.accuracy',
            metric_name='accuracy',
            metric_kind='accuracy',
            metric_unit='proportion',
            lower_is_better=False,
            score_type=ScoreType.continuous,
            min_score=0.0,
            max_score=1.0,
            metric_parameters={'aggregation_level': level, 'aggregation': 'micro'},
            additional_details={
                'n_items': str(agg.n),
                'n_correct': str(int(agg.correct)),
                'mean_input_tokens': f'{agg.in_tok / agg.n:.1f}' if agg.n else '0',
                'mean_output_tokens': f'{agg.out_tok / agg.n:.1f}' if agg.n else '0',
            },
        ),
        score_details=ScoreDetails(
            score=accuracy,
            details={'n_items': str(agg.n), 'n_correct': str(int(agg.correct))},
            uncertainty={'standard_error': {'value': se, 'method': 'analytic'},
                         'num_samples': agg.n},
        ),
    )


def build_log(model: str, task: str, subs: dict[str | None, Agg],
              eval_ts: str, retrieved_ts: str) -> tuple[EvaluationLog, str, str]:
    developer = model.split('/')[0] if '/' in model else 'unknown'
    model_slug = model.split('/')[-1]
    sanitized = model.replace('/', '_')
    real_subs = sorted(k for k in subs if k is not None)
    # If a benchmark has 0 or 1 distinct subtask, the overall == that lone subtask,
    # so emit only the overall (avoids byte-identical duplicate results, e.g. the
    # 17 tasks whose only subtask is "general"). Split into subtasks only when >1.
    results = [_result(task, None, subs[None])]
    if len(real_subs) > 1:
        for sub in real_subs:
            results.append(_result(task, sub, subs[sub]))
    log = EvaluationLog(
        schema_version=SCHEMA_VERSION,
        # id keyed on the (stable) evaluation time, so reruns are idempotent;
        # retrieved_timestamp records when WE built the record (now).
        evaluation_id=f'wild/{sanitized}/{task}/{eval_ts}',
        retrieved_timestamp=retrieved_ts,
        evaluation_timestamp=eval_ts,
        source_metadata=SourceMetadata(
            source_name=SOURCE_NAME,
            source_type='evaluation_run',
            source_organization_name=SOURCE_ORGANIZATION,
            source_organization_url=HF_DATASET_URL,
            evaluator_relationship=EvaluatorRelationship.third_party,
            additional_details={
                'dataset_url': HF_DATASET_URL,
                'paper_url': PAPER_URL,
                'note': 'Item-level evals run by Kensho with the Inspect AI '
                        'framework (WILD paper).',
            },
        ),
        eval_library=EvalLibrary(
            name='inspect_ai', version='unknown',
            additional_details={'note': 'Run with the Inspect AI framework (WILD paper).'},
        ),
        model_info=ModelInfo(
            name=model, id=model, developer=developer,
            additional_details={'wild_model_id': model},
        ),
        evaluation_results=results,
    )
    return log, developer, model_slug


# --------------------------------------------------------------------------- #
# instance-level (--include-instances)
# --------------------------------------------------------------------------- #

def _prompt_from_conversation(raw: str | None) -> str:
    """The input prompt = the user (and system) turns only. The `conversation`
    column also carries the assistant response, which must NOT go in input.raw
    (that would leak the answer and break cross-model sample hashing) — it's the
    model output and lives in output.raw."""
    if not raw:
        return ''
    try:
        msgs = json.loads(raw)
        parts = [m.get('content', '') for m in msgs
                 if m.get('role') in ('user', 'system') and m.get('content')]
        return '\n\n'.join(parts) if parts else ''
    except (ValueError, TypeError, AttributeError):
        return str(raw)


def _scorer_name(scores_json: str | None) -> str:
    """The Inspect scorer that produced the item score = the key of the scores
    JSON (e.g. 'match', 'choice', 'model_graded_qa')."""
    if scores_json:
        try:
            keys = list(json.loads(scores_json).keys())
            if keys:
                return str(keys[0])
        except (ValueError, TypeError, AttributeError):
            pass
    return 'unknown'


def _raw_output(scores_json: str | None, extracted: str) -> str:
    """The model's full generation lives in the scorer output (`scores.<scorer>.answer`);
    fall back to the extracted `answer` if absent."""
    if scores_json:
        try:
            for scorer in json.loads(scores_json).values():
                if isinstance(scorer, dict) and scorer.get('answer'):
                    return str(scorer['answer'])
        except (ValueError, TypeError, AttributeError):
            pass
    return str(extracted or '')


def make_instance(row: dict, evaluation_id: str, model: str,
                  multi_subtask: bool) -> InstanceLevelEvaluationLog:
    task = row['task']
    subtask = row['subtask'] if row['subtask'] not in (None, '') else '_'
    # attach to the leaf result when the benchmark is split by subtask, else the
    # single overall result (matches build_log's dedup so the FK always resolves).
    if multi_subtask:
        name, rid = f'wild.{task}.{subtask}', f'{task}::{subtask}'
    else:
        name, rid = f'wild.{task}', task
    score = float(row['score'] or 0)
    in_t, out_t = int(row['input_tokens'] or 0), int(row['output_tokens'] or 0)
    extracted = str(row.get('answer') or '')
    scorer = _scorer_name(row.get('scores'))
    prompt = _prompt_from_conversation(row.get('conversation'))
    reference = [str(row.get('target') or '')]
    sample_hash = hashlib.sha256((prompt + '\n' + '\n'.join(reference)).encode('utf-8')).hexdigest()
    return InstanceLevelEvaluationLog(
        schema_version=SCHEMA_VERSION,
        evaluation_id=evaluation_id,
        model_id=model,
        evaluation_name=name,
        evaluation_result_id=rid,
        sample_id=str(row['item_id']),
        sample_hash=sample_hash,
        interaction_type=InteractionType.single_turn,
        input=Input(raw=prompt, reference=reference),
        output=Output(raw=[_raw_output(row.get('scores'), extracted)]),
        answer_attribution=[AnswerAttributionItem(
            turn_idx=0, source='output.raw',
            extracted_value=extracted,
            extraction_method=scorer, is_terminal=True)],
        evaluation=Evaluation(score=score, is_correct=score == 1.0),
        token_usage=TokenUsage(input_tokens=in_t, output_tokens=out_t,
                               total_tokens=in_t + out_t),
        metadata={'stop_reason': str(row.get('stop_reason') or ''),
                  'subtask': str(subtask), 'scorer': scorer},
    )


def write_instances(parquet, limit_shards, models, agg_paths: dict, eval_ids: dict,
                    multi: set, max_instances: int | None) -> dict[tuple[str, str], int]:
    """Stream item rows into per-(model,task) `<stem>_samples.jsonl`. Returns counts."""
    counts: dict[tuple[str, str], int] = defaultdict(int)
    written = 0
    for batch in iter_batches(parquet, INSTANCE_COLUMNS, limit_shards):
        # group this row-group's rows by (model, task) to bound open handles
        buckets: dict[tuple[str, str], list[str]] = defaultdict(list)
        n = len(batch['model'])
        for i in range(n):
            model, task = batch['model'][i], batch['task'][i]
            if models and model not in models:
                continue
            if (model, task) not in agg_paths:
                continue
            if max_instances is not None and written >= max_instances:
                break
            row = {c: batch[c][i] for c in INSTANCE_COLUMNS}
            inst = make_instance(row, eval_ids[(model, task)], model,
                                 (model, task) in multi)
            buckets[(model, task)].append(
                json.dumps(inst.model_dump(mode='json', exclude_none=True),
                           ensure_ascii=False))
            written += 1
        for key, lines in buckets.items():
            sample_path = agg_paths[key].with_name(f'{agg_paths[key].stem}_samples.jsonl')
            with sample_path.open('a', encoding='utf-8') as fh:
                fh.write('\n'.join(lines) + '\n')
            counts[key] += len(lines)
        if max_instances is not None and written >= max_instances:
            break
    return counts


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #

def resolve_eval_timestamp(override: str | None) -> str:
    """When the evaluation was RUN. We don't have per-run times, so use the WILD
    dataset's HF lastModified as a stable proxy (override with --evaluation-timestamp)."""
    if override:
        return str(override)
    try:
        from huggingface_hub import HfApi
        info = HfApi().dataset_info(HF_REPO_ID)
        if info.lastModified:
            dt = info.lastModified
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return repr(dt.timestamp())
    except Exception:  # noqa: BLE001 - fall back to now
        pass
    return str(time.time())


def run(args: argparse.Namespace) -> int:
    models = set(args.models) if args.models else None
    # retrieved = when this record was created (now); evaluation = when WILD ran it.
    eval_ts = resolve_eval_timestamp(args.evaluation_timestamp)
    retrieved_ts = str(args.retrieved_timestamp) if args.retrieved_timestamp else str(time.time())
    print(f'evaluation_timestamp = {eval_ts} | retrieved_timestamp = {retrieved_ts}')

    groups = aggregate(args.parquet, args.limit_shards, models)
    print(f'aggregated {len(groups)} (model, benchmark) groups')

    agg_paths: dict[tuple[str, str], Path] = {}
    eval_ids: dict[tuple[str, str], str] = {}
    logs: dict[tuple[str, str], EvaluationLog] = {}
    for (model, task), subs in groups.items():
        log, developer, model_slug = build_log(model, task, subs, eval_ts, retrieved_ts)
        path = save_evaluation_log(log, args.output_dir, developer, model_slug)
        agg_paths[(model, task)] = path
        eval_ids[(model, task)] = log.evaluation_id
        logs[(model, task)] = log

    if args.include_instances:
        print('writing instance sidecars…')
        multi = {k for k, subs in groups.items()
                 if len([s for s in subs if s is not None]) > 1}
        counts = write_instances(args.parquet, args.limit_shards, models,
                                 agg_paths, eval_ids, multi, args.max_instances)
        for key, count in counts.items():
            if not count:
                continue
            path = agg_paths[key]
            sample_path = path.with_name(f'{path.stem}_samples.jsonl')
            checksum = hashlib.sha256(sample_path.read_bytes()).hexdigest()
            logs[key].detailed_evaluation_results = DetailedEvaluationResults(
                format=Format.jsonl, file_path=sample_path.name,
                hash_algorithm=HashAlgorithm.sha256, checksum=checksum,
                total_rows=count)
            path.write_text(
                logs[key].model_dump_json(indent=2, exclude_none=True),
                encoding='utf-8')
        print(f'wrote {sum(counts.values())} instance records')

    print(f'wrote {len(agg_paths)} aggregate EvaluationLog(s) -> {args.output_dir}')
    return len(agg_paths)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Convert kensho/WILD-raw to Every Eval Ever.')
    p.add_argument('--parquet', nargs='*', default=None,
                   help='Local parquet path(s); default fetches the HF shards.')
    p.add_argument('--output-dir', type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    p.add_argument('--limit-shards', type=int, default=None,
                   help='Only read the first N shards (for smoke runs).')
    p.add_argument('--models', nargs='*', default=None, help='Filter to these model ids.')
    p.add_argument('--include-instances', action='store_true',
                   help='Also write per-item `<uuid>_samples.jsonl` instance sidecars.')
    p.add_argument('--max-instances', type=int, default=None,
                   help='Cap total instance rows written (smoke runs).')
    p.add_argument('--retrieved-timestamp', default=None,
                   help='Override the record-creation epoch (default: now).')
    p.add_argument('--evaluation-timestamp', default=None,
                   help='Override when the eval ran (default: HF dataset lastModified).')
    return p.parse_args()


if __name__ == '__main__':
    written = run(parse_args())
    print(f'Wrote {written} WILD model×benchmark log(s).')
