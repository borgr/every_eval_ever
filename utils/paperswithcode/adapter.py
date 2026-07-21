#!/usr/bin/env python3
"""Convert Papers with Code evaluation results into Every Eval Ever records.

Data source:
- HF bucket ``nielsr/paperswithcode-backups`` -> nightly PostgreSQL custom-format
  dumps under ``postgres/*.dump`` (pg_dump ``-Fc``). Read with ``pgdumplib``
  (pure-python; no PostgreSQL server needed).

The relevant tables are:
- ``evaluations``  -> one row per (paper, task, dataset, model) leaderboard entry,
                      with a ``metrics`` jsonb of ``{metric_name: value}``.
- ``datasets``     -> the benchmark the eval ran on (source_data).
- ``tasks``        -> the task/category the benchmark belongs to.
- ``metrics``      -> metric definitions incl. ``direction`` (lower/higher_is_better).
- ``papers``       -> provenance (arXiv id / source url).

Shape (see the eee-dataset-conversion skill, reference/fields.md #shape):
- ``source_type = documentation`` -- PwC aggregates reported numbers; no raw outputs.
- aggregate ``.json`` only -- there is no per-item data.
- grain = one ``EvaluationLog`` per model; each ``evaluation_results[]`` entry is
  one (evaluation row x metric-in-jsonb) pair.

Usage:
    # from a dump already on disk (no network):
    uv run python -m utils.paperswithcode.adapter \
        --dump /tmp/pwc-raw/paperswithcode_hf_20260716_031511.dump \
        --dataset-slug eth3d-relative --dataset-slug re10k-2-view \
        --output-dir /tmp/eee-pwc

    # download the latest dump from the HF bucket first:
    uv run python -m utils.paperswithcode.adapter --output-dir data/paperswithcode

Then validate:
    python -m every_eval_ever validate /tmp/eee-pwc
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

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
    SourceDataPrivate,
    SourceDataUrl,
    SourceMetadata,
    StandardError,
    Uncertainty,
)
from every_eval_ever.helpers import (
    SCHEMA_VERSION,
    get_developer,
    sanitize_filename,
    save_evaluation_log,
)

SRC = 'paperswithcode'
PWC_SITE = 'https://paperswithcode.com'
DEFAULT_BUCKET = 'nielsr/paperswithcode-backups'
DEFAULT_OUTPUT_DIR = 'data/paperswithcode'

# Tables loaded in full to build lookups. ``papers`` is streamed separately and
# subsetted to only the paper ids referenced by the emitted evaluations.
LOOKUP_TABLES = ('datasets', 'tasks', 'metrics')

# A small slice that exercises every field decision:
#   eth3d-relative -> hf_dataset source, open + closed models, higher & lower
#                     is-better metrics, multi-metric rows.
#   re10k-2-view   -> url source, an unbounded metric (PSNR) and a lower-is-better
#                     metric (LPIPS), multi-metric rows.
SAMPLE_DATASET_SLUGS = ('eth3d-relative', 're10k-2-view')

# Vendored snapshot of the eval-card-registry's canonical metrics (bounds +
# direction + score_type). Regenerate with refresh_metric_snapshot.py. Committed
# so the adapter never needs the registry installed at runtime (matching every
# other adapter) while still sourcing bounds from the registry, not inventing them.
SNAPSHOT_PATH = Path(__file__).with_name('registry_metrics.json')

HF_MODEL_RE = re.compile(
    r'https?://huggingface\.co/(?!datasets/|spaces/)([^/\s]+)/([^/?#\s]+)'
)
HF_DATASET_RE = re.compile(
    r'https?://huggingface\.co/datasets/([^/\s]+/[^/?#\s]+)'
)


# ---------------------------------------------------------------------------
# String / value helpers
# ---------------------------------------------------------------------------


def stringify(value: Any) -> str:
    """Coerce a scalar/collection to a string for a `dict[str, str]` field."""
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if value is None:
        return 'null'
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(',', ':'))
    return str(value)


def stringify_details(details: dict[str, Any]) -> dict[str, str]:
    """Drop None values and stringify the rest (EEE string-maps forbid non-str)."""
    return {k: stringify(v) for k, v in details.items() if v is not None}


def coerce_bool(value: Any) -> bool | None:
    """Postgres dumps booleans as 't'/'f'; normalise to real bools."""
    if isinstance(value, bool):
        return value
    if value in ('t', 'true', 'True', '1'):
        return True
    if value in ('f', 'false', 'False', '0'):
        return False
    return None


def slugify(value: Any, fallback: str = 'unknown') -> str:
    raw = str(value if value not in (None, '') else fallback).strip().lower()
    raw = sanitize_filename(raw).replace('&', 'and')
    raw = re.sub(r'[\s_]+', '-', raw)
    raw = re.sub(r'[^a-z0-9.\-]+', '-', raw)
    raw = re.sub(r'-{2,}', '-', raw).strip('-')
    return raw or fallback


def snake(value: Any, fallback: str = 'unknown') -> str:
    return slugify(value, fallback).replace('-', '_').replace('.', '_')


def _to_float(text: str) -> float | None:
    """Parse a numeric string tolerating '%' and European decimal commas.

    PwC stores metric values as free text: '95.2', '95,2' (decimal comma),
    '1,234.5' (thousands sep), '30%'. A naive ``replace(',', '')`` corrupts
    '97,3' into 973, so the comma is only stripped when it is clearly a
    thousands separator.
    """
    s = str(text).strip().rstrip('%').strip()
    if not s:
        return None
    if ',' in s and '.' in s:
        s = s.replace(',', '')  # thousands separator: 1,234.5 -> 1234.5
    elif ',' in s:
        if re.fullmatch(r'-?\d+,\d{1,2}', s):
            s = s.replace(',', '.')  # decimal comma: 97,3 -> 97.3
        else:
            s = s.replace(',', '')  # thousands: 1,234 -> 1234
    try:
        return float(s)
    except ValueError:
        return None


def parse_metric_value(raw: Any) -> tuple[float | None, float | None]:
    """Return (score, standard_error). Handles 'mean +/- sd' uncertainty."""
    if raw is None:
        return None, None
    s = str(raw).strip()
    if not s:
        return None, None
    std_err = None
    for sep in ('±', '+/-', '+-'):
        if sep in s:
            left, _, right = s.partition(sep)
            s = left.strip()
            std_err = _to_float(right)
            break
    return _to_float(s), std_err


def dedupe(items: Iterable[Any]) -> list[Any]:
    seen: set[Any] = set()
    out: list[Any] = []
    for it in items:
        if it and it not in seen:
            out.append(it)
            seen.add(it)
    return out


# ---------------------------------------------------------------------------
# Field builders (pure -- operate on plain dicts, so tests need no DB / network)
# ---------------------------------------------------------------------------


def model_identity(
    model_name: Any, hf_model_url: Any
) -> tuple[str, str, str, str]:
    """Return (model_id, developer, model_slug, display_name).

    Prefers the HF url (keeps HF-true casing). Otherwise guesses the developer
    from the name and slugifies. NOTE: effort/mode tiers baked into PwC model
    names (e.g. 'GPT-5.5 Pro (xhigh)') are preserved in the slug rather than
    stripped -- collapsing them is the eval-card-registry's job (a separate PR).
    """
    display = str(model_name if model_name not in (None, '') else 'unknown')
    display = display.strip() or 'unknown'
    if hf_model_url:
        m = HF_MODEL_RE.match(str(hf_model_url).strip())
        if m:
            dev, mdl = m.group(1), m.group(2)
            return f'{dev}/{mdl}', dev, mdl, display
    dev = slugify(get_developer(display) or 'unknown')
    mdl = slugify(display, 'unknown')
    return f'{dev}/{mdl}', dev, mdl, display


def extract_hf_dataset_repo(hf_url: str) -> str | None:
    m = HF_DATASET_RE.match(str(hf_url).strip())
    return m.group(1) if m else None


def dataset_details(dataset: dict[str, Any]) -> dict[str, str]:
    return stringify_details(
        {
            'raw_dataset_id': dataset.get('id'),
            'pwc_dataset_slug': dataset.get('slug'),
            'pwc_dataset_url': f'{PWC_SITE}/dataset/{dataset.get("slug")}'
            if dataset.get('slug')
            else None,
            'license_name': dataset.get('license_name'),
            'license_url': dataset.get('license_url'),
            'introduced_year': dataset.get('introduced_year'),
            'paper_url': dataset.get('paper_url'),
        }
    )


def build_source_data(dataset: dict[str, Any]):
    """The DATASET the eval ran on. hf_dataset if an HF url exists, else url,
    else (never, in practice -- see the skill friction report) private/other."""
    name = dataset.get('name') or dataset.get('slug') or 'unknown'
    details = dataset_details(dataset)
    hf_url = dataset.get('hf_url')
    if hf_url:
        repo = extract_hf_dataset_repo(hf_url)
        if repo:
            return SourceDataHf(
                dataset_name=name,
                source_type='hf_dataset',
                hf_repo=repo,
                additional_details=details,
            )
    urls = dedupe(
        [
            dataset.get('url'),
            dataset.get('homepage'),
            dataset.get('paper_url'),
        ]
    )
    if dataset.get('slug'):
        urls.append(f'{PWC_SITE}/dataset/{dataset["slug"]}')
    urls = dedupe(urls)
    if urls:
        return SourceDataUrl(
            dataset_name=name,
            source_type='url',
            url=urls,
            additional_details=details,
        )
    return SourceDataPrivate(
        dataset_name=name, source_type='other', additional_details=details
    )


def _finite_bounds(lo: float, hi: float) -> tuple[float, float]:
    """Ensure a usable, VALID finite [min, max] (hi strictly > lo)."""
    lo, hi = float(lo), float(hi)
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def reconcile_scale(
    score: float, std_err: float | None, lo: float, hi: float, resolved: bool
) -> tuple[float, float | None, dict[str, Any]]:
    """Rescale a source value onto the canonical [lo, hi] scale.

    PwC reports proportion metrics (canonical [0,1]) as percent (0-100); this maps
    the score (and its std_err, same units) back to the canonical scale so a score
    is never outside its own bounds. Only the unambiguous percent->proportion case
    is rescaled; a value that still won't fit is left as-is and flagged rather than
    silently divided by a guessed factor.
    """
    detail: dict[str, Any] = {}
    if not resolved or lo <= score <= hi:
        return score, std_err, detail
    if 0.0 <= hi <= 1.0 and hi < score <= 100.0:
        detail['canonical_rescale_factor'] = 100.0
        score = score / 100.0
        if std_err is not None:
            std_err = std_err / 100.0
    else:
        detail['scale_anomaly'] = 'true'  # outside canonical range, not rescaled
    return score, std_err, detail


def _normalize_metric_key(name: Any) -> str:
    """Mirror the registry's `normalized` matcher: drop case + all separators."""
    return re.sub(r'[^a-z0-9]', '', str(name).lower())


@dataclass(frozen=True)
class ResolvedMetric:
    metric_id: str
    metric_kind: str
    lower_is_better: bool
    score_type: str
    min_score: float
    max_score: float
    resolved: bool  # True = came from the registry snapshot
    detail: dict[str, Any]


class MetricResolver:
    """Resolve a PwC metric name to canonical bounds/direction from the vendored
    registry snapshot (Tier 1). Unknown metrics are recorded and, unless the
    caller opts into a fallback, cause the run to fail closed (Tier 2); with
    ``allow_unresolved`` they get an observed-range proxy (Tier 3).

    Unbounded canonical bounds are ``null`` in the registry snapshot; they are
    emitted as ``+/-inf``, which serializes to the JSON string
    ``"Infinity"``/``"-Infinity"`` (see every_eval_ever#207). ``null`` means
    "not provided", never unbounded.
    """

    def __init__(
        self,
        pwc_directions: dict[str, str] | None = None,
        snapshot_path: str | Path = SNAPSHOT_PATH,
    ) -> None:
        data = json.loads(Path(snapshot_path).read_text(encoding='utf-8'))
        self._by_id = {m['id']: m for m in data['metrics']}
        self._index: dict[str, str] = {}
        for m in data['metrics']:
            keys = [m['id'], m.get('display_name'), *(m.get('aliases') or [])]
            for k in keys:
                if k:
                    self._index.setdefault(_normalize_metric_key(k), m['id'])
        self.pwc_directions = pwc_directions or {}
        # raw metric name -> set of dataset slugs it was seen on (for the report)
        self.unresolved: dict[str, set[str]] = {}
        # metric name -> count of results emitted with an unbounded (inf) bound
        self.unbounded_emitted: dict[str, int] = {}

    def _lookup(self, metric_name: str) -> dict[str, Any] | None:
        return self._by_id.get(self._index.get(_normalize_metric_key(metric_name)))

    def resolve(
        self,
        metric_name: str,
        obs_range: tuple[float, float],
        dataset_slug: str | None = None,
    ) -> ResolvedMetric:
        obs_min, obs_max = obs_range
        entry = self._lookup(metric_name)
        if entry is not None:
            score_type = entry.get('score_type') or 'continuous'
            detail: dict[str, Any] = {
                'bound_source': 'registry',
                'canonical_metric_id': entry['id'],
            }
            lo, hi = entry.get('min_score'), entry.get('max_score')
            # null bound in the registry == unbounded -> +/-inf (every_eval_ever#207
            # serializes these as the JSON string "Infinity"/"-Infinity").
            if lo is None:
                lo = float('-inf')
                detail['canonical_min'] = 'unbounded'
            if hi is None:
                hi = float('inf')
                detail['canonical_max'] = 'unbounded'
            if lo == float('-inf') or hi == float('inf'):
                self.unbounded_emitted[metric_name] = (
                    self.unbounded_emitted.get(metric_name, 0) + 1
                )
            return ResolvedMetric(
                metric_id=entry['id'],
                metric_kind=entry['id'],
                lower_is_better=bool(entry.get('lower_is_better')),
                score_type=score_type,
                min_score=float(lo),
                max_score=float(hi),
                resolved=True,
                detail=detail,
            )
        # unresolved -> record for the fail-closed report; return an observed proxy
        self.unresolved.setdefault(metric_name, set())
        if dataset_slug:
            self.unresolved[metric_name].add(dataset_slug)
        lo, hi = _finite_bounds(min(0.0, obs_min), obs_max)
        direction = self.pwc_directions.get(metric_name)
        return ResolvedMetric(
            metric_id=f'{SRC}.{snake(metric_name)}',
            metric_kind=snake(metric_name),
            lower_is_better=(direction == 'lower_is_better'),
            score_type='continuous',
            min_score=lo,
            max_score=hi,
            resolved=False,
            detail={
                'bound_source': 'observed_unresolved',
                'pwc_metric_direction': direction,
            },
        )


def build_metric_config(
    metric_name: str,
    resolved: ResolvedMetric,
    obs_range: tuple[float, float],
    metric_meta: dict[str, Any] | None,
) -> MetricConfig:
    meta = metric_meta or {}
    return MetricConfig(
        evaluation_description=meta.get('description'),
        metric_id=resolved.metric_id,
        metric_name=metric_name,
        metric_kind=resolved.metric_kind,
        # metric_unit left unset unless PwC declares a scale -- inferring it from
        # the value range mislabels physical-unit metrics (e.g. PSNR in dB).
        metric_unit=meta.get('scale') or None,
        lower_is_better=resolved.lower_is_better,
        score_type=ScoreType(resolved.score_type),
        min_score=resolved.min_score,
        max_score=resolved.max_score,
        additional_details=stringify_details(
            {
                **resolved.detail,
                'observed_min': obs_range[0],
                'observed_max': obs_range[1],
                'pwc_metric_full_name': meta.get('full_name'),
            }
        ),
    )


def score_details(
    ev: dict[str, Any],
    raw_value: Any,
    score: float,
    std_err: float | None,
    dataset: dict[str, Any],
    paper: dict[str, Any] | None,
    scale_detail: dict[str, Any] | None = None,
) -> ScoreDetails:
    unc = None
    if std_err is not None:
        unc = Uncertainty(
            standard_error=StandardError(value=std_err, method='reported')
        )
    paper = paper or {}
    arxiv_id = paper.get('arxiv_id')
    return ScoreDetails(
        score=score,
        uncertainty=unc,
        details=stringify_details(
            {
                **(scale_detail or {}),
                'raw_value': raw_value,
                'pwc_evaluation_id': ev.get('id'),
                'best_rank': ev.get('best_rank'),
                'best_metric': ev.get('best_metric'),
                'harness': _clean_harness(ev.get('harness')),
                'uses_additional_data': coerce_bool(
                    ev.get('uses_additional_data')
                ),
                'external': coerce_bool(ev.get('external')),
                'external_source_url': ev.get('external_source_url'),
                'source_url': ev.get('source_url'),
                'paper_arxiv_url': f'https://arxiv.org/abs/{arxiv_id}'
                if arxiv_id
                else None,
                'paper_title': paper.get('title'),
            }
        ),
    )


def _clean_harness(harness: Any) -> str | None:
    """PwC 'harness' is often an agent scaffold or 'Not reported' -- not a classic
    eval library. Normalise obvious non-values to None."""
    if not harness:
        return None
    text = str(harness).strip()
    if text.lower() in ('not reported', 'none', 'n/a', 'unknown', 'pool'):
        return None
    return text


def build_results(
    ev: dict[str, Any],
    dataset: dict[str, Any],
    task: dict[str, Any] | None,
    resolver: MetricResolver,
    metric_ranges: dict[str, tuple[float, float]],
    metric_meta: dict[str, dict[str, Any]],
    paper: dict[str, Any] | None,
) -> list[EvaluationResult]:
    """Fan one evaluation row out to one EvaluationResult per jsonb metric."""
    try:
        metrics = json.loads(ev['metrics']) if ev.get('metrics') else {}
    except (TypeError, ValueError):
        metrics = {}
    if not isinstance(metrics, dict) or not metrics:
        return []

    ds_slug = dataset.get('slug') or slugify(dataset.get('name'))
    task_slug = (task or {}).get('slug') or 'unknown-task'
    eval_name = f'{SRC}.{snake(task_slug)}.{snake(ds_slug)}'
    src_data = build_source_data(dataset)
    ts = ev.get('evaluated_on') or (str(ev.get('created_at') or '')[:10] or None)

    results: list[EvaluationResult] = []
    for mname, raw in metrics.items():
        score, std_err = parse_metric_value(raw)
        if score is None:
            continue
        obs_range = metric_ranges.get(mname, (score, score))
        resolved = resolver.resolve(mname, obs_range, ds_slug)
        score, std_err, scale_detail = reconcile_scale(
            score, std_err, resolved.min_score, resolved.max_score, resolved.resolved
        )
        results.append(
            EvaluationResult(
                evaluation_result_id=f'{SRC}.{ev.get("id")}.{snake(mname)}',
                evaluation_name=eval_name,
                source_data=src_data,
                evaluation_timestamp=str(ts) if ts else None,
                metric_config=build_metric_config(
                    mname, resolved, obs_range, metric_meta.get(mname)
                ),
                score_details=score_details(
                    ev, raw, score, std_err, dataset, paper, scale_detail
                ),
            )
        )
    return results


def build_source_metadata(dump_version: str) -> SourceMetadata:
    return SourceMetadata(
        source_name='Papers with Code',
        source_type='documentation',
        source_organization_name='Papers with Code',
        source_organization_url=PWC_SITE,
        # A leaderboard aggregating reported numbers is third_party wrt the model
        # developer, even when a score was self-reported to it (see fields.md).
        evaluator_relationship=EvaluatorRelationship.third_party,
        additional_details={
            'source_role': 'aggregator',
            'source_bucket': DEFAULT_BUCKET,
            'dump_version': dump_version,
            'note': (
                'Scores aggregated by Papers with Code from papers and external '
                'leaderboards; not re-run by this adapter.'
            ),
        },
    )


def build_model_info(
    model_id: str, developer: str, display_name: str, ev: dict[str, Any]
) -> ModelInfo:
    return ModelInfo(
        name=display_name,
        id=model_id,
        developer=developer,
        additional_details=stringify_details(
            {
                'raw_model_name': display_name,
                'hf_model_url': ev.get('hf_model_url'),
                'num_parameters': ev.get('num_parameters'),
                'is_open': coerce_bool(ev.get('is_open')),
            }
        ),
    )


@dataclass(frozen=True)
class LogBundle:
    log: EvaluationLog
    developer: str
    model: str


def build_logs(
    evaluations: Iterable[dict[str, Any]],
    datasets_by_id: dict[Any, dict[str, Any]],
    tasks_by_id: dict[Any, dict[str, Any]],
    resolver: MetricResolver,
    metric_ranges: dict[str, tuple[float, float]],
    metric_meta: dict[str, dict[str, Any]],
    papers_by_id: dict[Any, dict[str, Any]],
    dump_version: str,
    retrieved_ts: str,
) -> list[LogBundle]:
    """Group evaluation rows by canonical model id into one log per model."""
    groups: dict[str, list[EvaluationResult]] = defaultdict(list)
    infos: dict[str, ModelInfo] = {}
    harnesses: dict[str, set[str]] = defaultdict(set)
    devmodel: dict[str, tuple[str, str]] = {}

    for ev in evaluations:
        dataset = datasets_by_id.get(ev.get('dataset_id'))
        if dataset is None:
            continue
        task = tasks_by_id.get(ev.get('task_id'))
        paper = papers_by_id.get(ev.get('paper_id'))
        results = build_results(
            ev, dataset, task, resolver, metric_ranges, metric_meta, paper
        )
        if not results:
            continue
        model_id, developer, model_slug, display = model_identity(
            ev.get('model_name'), ev.get('hf_model_url')
        )
        groups[model_id].extend(results)
        harness = _clean_harness(ev.get('harness'))
        if harness:
            harnesses[model_id].add(harness)
        if model_id not in infos:
            infos[model_id] = build_model_info(
                model_id, developer, display, ev
            )
            devmodel[model_id] = (developer, model_slug)

    bundles: list[LogBundle] = []
    for model_id, results in sorted(groups.items()):
        developer, model_slug = devmodel[model_id]
        # eval_library is reserved for the eval *harness* (inspect_ai/lm-eval/helm).
        # PwC's `harness` column is usually an agent scaffold (SWE-agent, OpenHands)
        # or "Not reported" -- NOT a harness -- so eval_library stays 'unknown' and
        # any scaffold is recorded in additional_details instead.
        harness_set = harnesses.get(model_id, set())
        eval_lib_details = (
            {'pwc_harness': ', '.join(sorted(harness_set))} if harness_set else None
        )
        log = EvaluationLog(
            schema_version=SCHEMA_VERSION,
            # STABLE anchor: model id + dump version -> idempotent per dump, never `now`.
            evaluation_id=f'{SRC}/{model_id.replace("/", "_")}/{dump_version}',
            retrieved_timestamp=retrieved_ts,
            source_metadata=build_source_metadata(dump_version),
            eval_library=EvalLibrary(
                name='unknown',
                version='unknown',
                additional_details=eval_lib_details,
            ),
            model_info=infos[model_id],
            evaluation_results=sorted(
                results, key=lambda r: r.evaluation_result_id or ''
            ),
        )
        bundles.append(
            LogBundle(log=log, developer=developer, model=model_slug)
        )
    return bundles


# ---------------------------------------------------------------------------
# Dump IO (pgdumplib) -- kept out of the pure builders so tests need no DB
# ---------------------------------------------------------------------------


def _parse_columns(create_defn: str) -> list[str]:
    """Extract column names (in order) from a CREATE TABLE statement."""
    body = create_defn.split('(', 1)[1]
    cols: list[str] = []
    for line in body.splitlines():
        line = line.strip().rstrip(',')
        if not line or line.startswith('CONSTRAINT') or line.startswith(')'):
            continue
        cols.append(line.split()[0])
    return cols


def load_dump(dump_path: str | Path):
    import pgdumplib

    return pgdumplib.load(str(dump_path))


def _columns_for(dump, table: str) -> list[str]:
    for e in dump.entries:
        if e.desc == 'TABLE' and e.tag == table:
            return _parse_columns(e.defn)
    raise KeyError(f'table public.{table} not found in dump')


def table_rows(dump, table: str) -> Iterator[dict[str, Any]]:
    cols = _columns_for(dump, table)
    for row in dump.table_data('public', table):
        yield dict(zip(cols, row))


def _iter_metric_values(metrics_json: Any) -> Iterator[tuple[str, float]]:
    try:
        metrics = json.loads(metrics_json) if metrics_json else {}
    except (TypeError, ValueError):
        return
    if not isinstance(metrics, dict):
        return
    for name, raw in metrics.items():
        val, _ = parse_metric_value(raw)
        if val is not None:
            yield name, val


def scan_evaluations(
    dump,
    dataset_ids: set[Any] | None,
    limit: int | None,
) -> tuple[list[dict[str, Any]], dict[str, tuple[float, float]]]:
    """Single pass over ``evaluations``: accumulate per-metric observed ranges
    over the WHOLE dump (stable bounds, slice-independent) while collecting the
    filtered rows to emit."""
    ranges: dict[str, list[float]] = defaultdict(lambda: [float('inf'), float('-inf')])
    selected: list[dict[str, Any]] = []
    for ev in table_rows(dump, 'evaluations'):
        for name, val in _iter_metric_values(ev.get('metrics')):
            r = ranges[name]
            r[0] = min(r[0], val)
            r[1] = max(r[1], val)
        if dataset_ids is not None and ev.get('dataset_id') not in dataset_ids:
            continue
        if limit is not None and len(selected) >= limit:
            continue
        selected.append(ev)
    metric_ranges = {k: (lo, hi) for k, (lo, hi) in ranges.items()}
    return selected, metric_ranges


def read_papers_subset(dump, paper_ids: set[Any]) -> dict[Any, dict[str, Any]]:
    if not paper_ids:
        return {}
    want = {str(p) for p in paper_ids}
    out: dict[Any, dict[str, Any]] = {}
    for row in table_rows(dump, 'papers'):
        if str(row.get('id')) in want:
            out[row['id']] = {
                'arxiv_id': row.get('arxiv_id'),
                'title': row.get('title'),
                'source_url': row.get('source_url'),
            }
    return out


def dump_version_from_path(dump_path: str | Path) -> str:
    m = re.search(r'(\d{8})(?:_\d+)?', Path(dump_path).name)
    return m.group(1) if m else Path(dump_path).stem


# ---------------------------------------------------------------------------
# HF bucket download
# ---------------------------------------------------------------------------


def latest_dump_remote_path(bucket: str) -> str:
    from huggingface_hub import HfApi

    api = HfApi()
    dumps = [
        f.path
        for f in api.list_bucket_tree(bucket)
        if getattr(f, 'path', '').endswith('.dump')
    ]
    if not dumps:
        raise SystemExit(f'no .dump files found in bucket {bucket}')
    return sorted(dumps)[-1]


def download_dump(bucket: str, remote_path: str, dest_dir: Path) -> Path:
    from huggingface_hub import HfApi

    dest_dir.mkdir(parents=True, exist_ok=True)
    local = dest_dir / Path(remote_path).name
    if local.exists():
        print(f'reusing cached dump {local}')
        return local
    print(f'downloading {bucket}:{remote_path} -> {local}')
    HfApi().download_bucket_files(
        bucket, [(remote_path, str(local))], raise_on_missing_files=True
    )
    return local


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description='Convert Papers with Code dumps to Every Eval Ever format.'
    )
    ap.add_argument(
        '--dump',
        type=Path,
        help='Path to a local .dump file. If omitted, a dump is fetched from '
        'the HF bucket.',
    )
    ap.add_argument(
        '--bucket',
        default=DEFAULT_BUCKET,
        help=f'HF bucket to fetch the dump from (default: {DEFAULT_BUCKET}).',
    )
    ap.add_argument(
        '--remote-path',
        help='Specific postgres/*.dump path in the bucket (default: latest).',
    )
    ap.add_argument(
        '--raw-dir',
        type=Path,
        default=Path('/tmp/pwc-raw'),
        help='Where to download the dump (default: /tmp/pwc-raw).',
    )
    ap.add_argument(
        '--dataset-slug',
        action='append',
        dest='dataset_slugs',
        help='Restrict to this dataset slug (repeatable). Defaults to a small '
        'representative sample; pass --all to convert everything.',
    )
    ap.add_argument(
        '--all',
        action='store_true',
        help='Convert every dataset (overrides the default sample slice).',
    )
    ap.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Cap the number of evaluation rows emitted (after filtering).',
    )
    ap.add_argument(
        '--allow-unresolved',
        action='store_true',
        help='Emit metrics that do not resolve in the registry snapshot using '
        'observed-range bounds (labelled). Without this the run FAILS CLOSED on '
        'any unresolved metric so CI never ships un-vetted bounds.',
    )
    ap.add_argument(
        '--output-dir',
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help=f'Output directory (default: {DEFAULT_OUTPUT_DIR}).',
    )
    return ap.parse_args()


def _report_unresolved(unresolved: dict[str, set[str]]) -> str:
    lines = [
        f'  - {name!r} (on {sorted(ds)})' for name, ds in sorted(unresolved.items())
    ]
    return (
        f'{len(unresolved)} metric(s) do not resolve in the registry snapshot '
        f'({SNAPSHOT_PATH.name}):\n'
        + '\n'.join(lines)
        + '\n\nAdd them to the eval-card-registry (see its `registry-entity-aliases` '
        'skill) with paper-defined min/max/direction, refresh the snapshot '
        '(refresh_metric_snapshot.py), and re-run — or pass --allow-unresolved to '
        'emit them now with observed-range bounds (labelled bound_source).'
    )


def run(args: argparse.Namespace) -> int:
    if args.dump is not None:
        dump_path = args.dump
    else:
        remote = args.remote_path or latest_dump_remote_path(args.bucket)
        dump_path = download_dump(args.bucket, remote, args.raw_dir)

    dump_version = dump_version_from_path(dump_path)
    retrieved_ts = str(time.time())

    print(f'loading dump {dump_path} ...')
    dump = load_dump(dump_path)

    datasets_by_id = {d['id']: d for d in table_rows(dump, 'datasets')}
    tasks_by_id = {t['id']: t for t in table_rows(dump, 'tasks')}
    metric_dir: dict[str, str] = {}
    metric_meta: dict[str, dict[str, Any]] = {}
    for m in table_rows(dump, 'metrics'):
        metric_dir[m['name']] = m.get('direction')
        metric_meta[m['name']] = m

    if args.all:
        dataset_ids: set[Any] | None = None
    else:
        slugs = set(args.dataset_slugs or SAMPLE_DATASET_SLUGS)
        dataset_ids = {
            d['id']
            for d in datasets_by_id.values()
            if d.get('slug') in slugs
        }
        missing = slugs - {
            datasets_by_id[i].get('slug') for i in dataset_ids
        }
        if missing:
            print(f'warning: dataset slug(s) not found: {sorted(missing)}')

    selected, metric_ranges = scan_evaluations(dump, dataset_ids, args.limit)
    print(f'selected {len(selected)} evaluation row(s)')

    paper_ids = {ev.get('paper_id') for ev in selected if ev.get('paper_id')}
    papers_by_id = read_papers_subset(dump, paper_ids)

    resolver = MetricResolver(pwc_directions=metric_dir)
    bundles = build_logs(
        selected,
        datasets_by_id,
        tasks_by_id,
        resolver,
        metric_ranges,
        metric_meta,
        papers_by_id,
        dump_version,
        retrieved_ts,
    )

    # Fail closed: never write un-vetted bounds unless explicitly allowed.
    if resolver.unresolved and not args.allow_unresolved:
        raise SystemExit('ERROR: ' + _report_unresolved(resolver.unresolved))

    for bundle in bundles:
        save_evaluation_log(
            bundle.log, args.output_dir, bundle.developer, bundle.model
        )
    total_results = sum(len(b.log.evaluation_results) for b in bundles)
    print(
        f'wrote {len(bundles)} model log(s), {total_results} result(s) '
        f'-> {args.output_dir}'
    )
    if resolver.unresolved:
        print(
            f'WARNING: {len(resolver.unresolved)} metric(s) used observed-range '
            f'fallback (--allow-unresolved): '
            f'{sorted(resolver.unresolved)}. Upstream them to the registry.'
        )
    if resolver.unbounded_emitted:
        print(
            f'NOTE: {sum(resolver.unbounded_emitted.values())} result(s) for '
            f'metric(s) {sorted(resolver.unbounded_emitted)} emitted with '
            f'unbounded (inf) bounds, serialized as "Infinity" '
            f'(every_eval_ever#207).'
        )
    return len(bundles)


if __name__ == '__main__':
    run(parse_args())
    # then validate:  python -m every_eval_ever validate <output-dir>
