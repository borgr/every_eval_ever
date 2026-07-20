#!/usr/bin/env python3
"""Convert the Open Medical-LLM Leaderboard results into Every Eval Ever aggregate logs.

Data source (HuggingFace dataset): openlifescienceai/results
  Layout: <developer>/<model>/results_*.json  (lm-evaluation-harness output format)
  Backs the Space: openlifescienceai/open_medical_llm_leaderboard

One EvaluationLog per model (developer/model), with one EvaluationResult per medical
benchmark (accuracy, proportion 0-1, higher is better). Aggregates only; no per-item data.

Run from the EEE repo dir:
    uv run python -m utils.open_medical_llm.adapter --output-dir /tmp/eee-omll [--limit N]
    uv run python -m every_eval_ever validate /tmp/eee-omll
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone

from every_eval_ever.eval_types import (
    EvalLibrary,
    EvaluationLog,
    EvaluationResult,
    EvaluatorRelationship,
    MetricConfig,
    ModelInfo,
    ScoreDetails,
    ScoreType,
    SourceDataHf,
    SourceMetadata,
    StandardError,
    Uncertainty,
)
from every_eval_ever.helpers import (
    SCHEMA_VERSION,
    save_evaluation_log,
)

REPO = "openlifescienceai/results"
RESOLVE = "https://huggingface.co/datasets/openlifescienceai/results/resolve/main/"
TREE = "https://huggingface.co/api/datasets/openlifescienceai/results/tree/main?recursive=true"
LEADERBOARD_SPACE = "https://huggingface.co/spaces/openlifescienceai/open_medical_llm_leaderboard"
SRC = "open-medical-llm-leaderboard"

# task name -> (human display name, verified HF dataset repo)
TASKS = {
    "medmcqa": ("MedMCQA", "openlifescienceai/medmcqa"),
    "medqa_4options": ("MedQA (USMLE, 4 options)", "openlifescienceai/MedQA-USMLE-4-options-hf"),
    "pubmedqa": ("PubMedQA", "openlifescienceai/pubmedqa"),
    "mmlu_anatomy": ("MMLU: Anatomy", "openlifescienceai/mmlu_anatomy"),
    "mmlu_clinical_knowledge": ("MMLU: Clinical Knowledge", "openlifescienceai/mmlu_clinical_knowledge"),
    "mmlu_college_biology": ("MMLU: College Biology", "openlifescienceai/mmlu_college_biology"),
    "mmlu_college_medicine": ("MMLU: College Medicine", "openlifescienceai/mmlu_college_medicine"),
    "mmlu_medical_genetics": ("MMLU: Medical Genetics", "openlifescienceai/mmlu_medical_genetics"),
    "mmlu_professional_medicine": ("MMLU: Professional Medicine", "openlifescienceai/mmlu_professional_medicine"),
}

TS_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})[ _](\d{2})[:_-](\d{2})[:_-](\d{2})")


def stringify(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (dict, list)):
        return json.dumps(v, sort_keys=True, separators=(",", ":"))
    return str(v)


def clean_details(d: dict) -> dict:
    return {k: stringify(v) for k, v in d.items() if v is not None}


def parse_ts(path: str):
    m = TS_RE.search(path)
    if not m:
        return None
    y, mo, da, h, mi, s = map(int, m.groups())
    try:
        return datetime(y, mo, da, h, mi, s, tzinfo=timezone.utc)
    except ValueError:
        return None


def fetch_json(path: str) -> dict:
    url = RESOLVE + urllib.parse.quote(path)
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read())


def list_result_files() -> list[str]:
    with urllib.request.urlopen(TREE, timeout=60) as r:
        tree = json.loads(r.read())
    out = []
    for x in tree:
        if x.get("type") != "file":
            continue
        p = x["path"]
        segs = p.split("/")
        # skip hidden dirs (e.g. .ipynb_checkpoints) and jupyter checkpoint files
        if any(s.startswith(".") for s in segs) or segs[-1].endswith("-checkpoint.json"):
            continue
        if p.endswith(".json") and segs[-1].startswith("results_"):
            out.append(p)
    return out


def latest_per_model(paths: list[str]) -> tuple[dict[str, str], list[str]]:
    """Group 3-segment ``developer/model/results_*.json`` paths, latest file per model.

    Root-level 2-segment baselines (e.g. ``GPT-4/results_*.json``) are returned
    separately and skipped: they are hand-curated closed-model paper numbers
    (bare ``acc`` only, no ``acc_stderr``/``model_args``/``bootstrap_iters``),
    NOT lm-evaluation-harness runs, so their provenance differs from the rest.
    """
    by_model: dict[str, list[str]] = defaultdict(list)
    baselines: list[str] = []
    for p in paths:
        parts = p.split("/")
        if len(parts) < 3:
            baselines.append(p)
            continue
        by_model["/".join(parts[:2])].append(p)
    chosen = {}
    for model, files in by_model.items():
        files.sort(key=lambda p: (parse_ts(p) or datetime.min.replace(tzinfo=timezone.utc), p))
        chosen[model] = files[-1]
    return chosen, baselines


def make_result(task: str, metrics: dict, eval_ts_iso: str | None) -> EvaluationResult | None:
    acc = metrics.get("acc,none")
    if acc is None:
        return None
    display, hf_repo = TASKS[task]

    stderr = metrics.get("acc_stderr,none")
    uncertainty = None
    if stderr is not None:
        uncertainty = Uncertainty(standard_error=StandardError(value=float(stderr)))

    score_details = ScoreDetails(
        score=float(acc),
        details=clean_details(
            {
                "raw_metric_key": "acc,none",
                "acc_norm": metrics.get("acc_norm,none"),
                "acc_norm_stderr": metrics.get("acc_norm_stderr,none"),
                "harness_alias": metrics.get("alias"),
            }
        ),
        uncertainty=uncertainty,
    )

    return EvaluationResult(
        evaluation_result_id=f"{SRC}.{task}",
        evaluation_name=f"{SRC}.{task}",
        evaluation_timestamp=eval_ts_iso,
        source_data=SourceDataHf(
            dataset_name=display,
            source_type="hf_dataset",
            hf_repo=hf_repo,
        ),
        metric_config=MetricConfig(
            evaluation_description=(
                f"Accuracy on the {display} medical QA benchmark as reported by the "
                "Open Medical-LLM Leaderboard."
            ),
            metric_id=f"{SRC}.{task}.accuracy",
            metric_name="accuracy",
            metric_kind="accuracy",
            metric_unit="proportion",
            lower_is_better=False,
            score_type=ScoreType.continuous,
            min_score=0.0,
            max_score=1.0,
        ),
        score_details=score_details,
    )


def make_log(model_repo: str, obj: dict, path: str, retrieved_ts: str) -> tuple[EvaluationLog, str, str] | None:
    developer, model = model_repo.split("/", 1)
    config = obj.get("config", {}) or {}
    results = obj.get("results", {}) or {}

    ev_results = []
    for task in TASKS:
        md = results.get(task)
        if isinstance(md, dict):
            r = make_result(task, md, None)  # per-result ts filled below
            if r is not None:
                ev_results.append(r)
    if not ev_results:
        return None

    eval_dt = parse_ts(path)
    eval_ts_iso = eval_dt.isoformat() if eval_dt else None
    for r in ev_results:
        r.evaluation_timestamp = eval_ts_iso

    if eval_dt:
        ts_token = str(int(eval_dt.timestamp()))
    else:
        # No parseable timestamp in the filename: derive a STABLE token from the
        # result file path (never `now`), so evaluation_id stays idempotent across
        # re-runs. evaluation_timestamp is left None (the run time is unknown).
        ts_token = re.sub(r"[^0-9A-Za-z._-]+", "-", path.rsplit("/", 1)[-1].removesuffix(".json"))
    model_id = model_repo  # already an HF repo id developer/model (canonical per registry S1)

    model_info = ModelInfo(
        name=config.get("model_name") or model_repo,
        id=model_id,
        developer=developer,
        additional_details=clean_details(
            {
                "model_sha": config.get("model_sha"),
                "model_dtype": config.get("model_dtype"),
                "model_args": config.get("model_args"),
                "model_num_parameters": config.get("model_num_parameters"),
                "model_revision": config.get("model_revision"),
            }
        ),
    )

    log = EvaluationLog(
        schema_version=SCHEMA_VERSION,
        evaluation_id=f"{SRC}/{model_id.replace('/', '_')}/{ts_token}",
        evaluation_timestamp=eval_ts_iso,
        retrieved_timestamp=retrieved_ts,
        source_metadata=SourceMetadata(
            source_name="Open Medical-LLM Leaderboard",
            source_type="documentation",
            source_organization_name="Open Life Science AI",
            source_organization_url=LEADERBOARD_SPACE,
            evaluator_relationship=EvaluatorRelationship.third_party,
            additional_details={
                "source_role": "aggregator",
                "results_dataset": REPO,
                "source_result_file": path,
            },
        ),
        eval_library=EvalLibrary(
            name="lm-evaluation-harness",
            version="unknown",
            additional_details={
                "inferred_from": "result format (acc,none / acc_stderr,none / bootstrap_iters / fewshot_seed)"
            },
        ),
        model_info=model_info,
        evaluation_results=sorted(ev_results, key=lambda r: r.evaluation_name),
    )
    return log, developer, model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="data/open-medical-llm")
    ap.add_argument("--limit", type=int, default=None, help="Process only the first N models.")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    retrieved_ts = str(time.time())
    paths = list_result_files()
    chosen, baselines = latest_per_model(paths)
    models = sorted(chosen)
    if args.limit:
        models = models[: args.limit]
    print(f"Models to process: {len(models)}")
    print(
        f"Skipped {len(baselines)} hand-curated baseline entries (different provenance): "
        + ", ".join(sorted(p.split('/')[0] for p in baselines))
    )

    def worker(model_repo: str):
        try:
            obj = fetch_json(chosen[model_repo])
        except Exception as e:  # noqa: BLE001
            return ("ERR", model_repo, str(e))
        built = make_log(model_repo, obj, chosen[model_repo], retrieved_ts)
        if built is None:
            return ("SKIP", model_repo, "no medical results")
        return ("OK", model_repo, built)

    written = errors = skipped = 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for status, model_repo, payload in ex.map(worker, models):
            if status == "OK":
                log, developer, model = payload
                save_evaluation_log(log, args.output_dir, developer, model)
                written += 1
            elif status == "SKIP":
                skipped += 1
            else:
                errors += 1
                print(f"  ERROR {model_repo}: {payload}")
    print(f"Wrote {written} logs; skipped {skipped}; errors {errors}. -> {args.output_dir}")
    return written


if __name__ == "__main__":
    main()
