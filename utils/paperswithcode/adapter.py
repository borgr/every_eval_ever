#!/usr/bin/env python3
"""Convert Papers with Code evaluation results into Every Eval Ever records.

Data source:
- HF bucket ``huggingface/paperswithcode-backups`` -> nightly PostgreSQL custom-format
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
import math
import re
import shutil
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
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
)
from every_eval_ever.helpers import (
    SCHEMA_VERSION,
    get_developer,
    sanitize_filename,
    save_evaluation_log,
)

SRC = 'paperswithcode'
PWC_SITE = 'https://paperswithcode.com'
DEFAULT_BUCKET = 'huggingface/paperswithcode-backups'
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

    PwC stores metric values as free text: '95.2', '95,2'/'0,991'/'97,345'
    (decimal comma), '1,234.5'/'1,234,567' (thousands separator), '30%'. The old
    rule only treated 1-2 digits after a comma as a decimal, so '0,991' became
    991 and '97,345' became 97345 -- precisions that are routine for eval metrics.
    A *single* comma is now read as a decimal separator (a bare thousands-grouped
    metric score is vanishingly rare); a comma alongside a '.', or several commas,
    is a thousands separator and stripped. Non-finite inputs ('NaN', 'Infinity',
    'inf') are rejected -- a score must be a finite real number, not a bound.
    """
    s = str(text).strip().rstrip('%').strip()
    if not s:
        return None
    if (',' in s and '.' in s) or s.count(',') > 1:
        s = s.replace(',', '')   # thousands: 1,234.5 -> 1234.5 ; 1,234,567 -> 1234567
    elif ',' in s:
        s = s.replace(',', '.')  # decimal comma: 97,3 -> 97.3 ; 0,991 -> 0.991
    try:
        val = float(s)
    except ValueError:
        return None
    return val if math.isfinite(val) else None


def parse_metric_value(raw: Any) -> tuple[float | None, str | None]:
    """Return (score, uncertainty_text).

    A 'mean +/- sd' value yields the numeric mean plus the raw right-hand token
    as *text*. PwC's '±' does not identify the spread as a standard error, a
    standard deviation, or a CI half-width, so the caller keeps it verbatim rather
    than coercing it into a typed Uncertainty (every_eval_ever#209 review).
    """
    if raw is None:
        return None, None
    s = str(raw).strip()
    if not s:
        return None, None
    unc_text: str | None = None
    for sep in ('±', '+/-', '+-'):
        if sep in s:
            left, _, right = s.partition(sep)
            s = left.strip()
            unc_text = right.strip() or None
            break
    return _to_float(s), unc_text


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
    score: float,
    lo: float,
    hi: float,
    resolved: bool,
    group_repr: float | None = None,
) -> tuple[float, dict[str, Any]]:
    """Rescale a source value onto the canonical [lo, hi] scale.

    PwC reports proportion metrics (canonical [0,1]) sometimes as percent (0-100),
    inconsistently even within one leaderboard. The reporting scale is a property
    of the whole (metric, dataset) group, so we decide it ONCE from a robust
    representative of that group -- its median (``group_repr``) -- rather than from
    each score in isolation. Per-score inference is fragile both ways: it would
    rescale a lone in-range value sitting in an otherwise-percent board (a ``1.0``
    that means 1%), and would silently "fix" a lone out-of-range value in an
    otherwise-proportion board (a mis-entered ``95``) instead of flagging it. With
    no group context supplied, ``group_repr`` falls back to the score itself (the
    original per-score behaviour, still correct for a singleton group).

    Only the unambiguous percent->proportion case (canonical ``hi <= 1`` with the
    group centred in ``(hi, 100]``) is rescaled. A representative above 100, or a
    score still outside the canonical range after applying the group scale, is
    flagged (``scale_anomaly``) and left as-is rather than divided by a guessed
    factor (every_eval_ever#209 review, Q2/Erotemic).
    """
    detail: dict[str, Any] = {}
    if not resolved:
        return score, detail
    ref = group_repr if group_repr is not None else score
    if math.isfinite(hi) and hi <= 1.0 and ref > hi:
        if ref <= 100.0:
            score = score / 100.0
            detail['canonical_rescale_factor'] = 100.0
            detail['rescale_basis'] = (
                'group_median' if group_repr is not None else 'single_score'
            )
        else:
            # group centred too high to be percent -> scale is unknown, don't guess
            detail['scale_anomaly'] = 'group_representative_above_percent'
    if math.isfinite(lo) and math.isfinite(hi) and not (lo <= score <= hi):
        detail.setdefault('scale_anomaly', 'score_outside_canonical_range')
    # Reproducibility: whenever a scale DECISION was made (rescale or anomaly),
    # record the representative it was based on, so the decision can be re-derived
    # and audited across dumps (every_eval_ever#209 review, Q2/Erotemic). The
    # no-op path (nothing decided) stays an empty detail.
    if detail:
        detail['scale_group_repr'] = ref
    return score, detail


def _normalize_metric_key(name: Any) -> str:
    """Mirror the registry's `normalized` matcher: drop case + all separators."""
    return re.sub(r'[^a-z0-9]', '', str(name).lower())


def _metadata_kind_confidence(meta: Any) -> tuple[str | None, str | None]:
    """Pull (kind, confidence) out of a registry metric's `metadata` field.

    The seed stores it as a JSON *string* (e.g. '{"kind": "real", "confidence":
    "high"}'); the snapshot may keep it as that string or as a parsed object.
    Best-effort: a malformed/absent value yields (None, None), never raises.
    """
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except (ValueError, TypeError):
            return None, None
    if not isinstance(meta, dict):
        return None, None
    kind = meta.get('kind')
    confidence = meta.get('confidence')
    return (
        str(kind) if kind is not None else None,
        str(confidence) if confidence is not None else None,
    )


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
        # The exact registry commit these bounds came from (surfaced per metric so
        # a downstream consumer can re-check a value if the registry moves).
        self.registry_revision = (data.get('_meta') or {}).get('registry_revision')
        # Two indices, EXACT-first. A spelling (id/display_name/alias) maps to the
        # set of canonical ids that use it. Exact (case-insensitive) match wins;
        # only if that misses do we fall back to the normalized key (case +
        # separators dropped). A key that maps to >1 id is an unresolvable
        # COLLISION -- we refuse to silently pick one (the old `setdefault` index
        # let the first-seen spelling win, so 'CLIP-IQA'/'CLIPIQA+' could steal
        # each other's id). (every_eval_ever#209 review, mrshu)
        self._exact: dict[str, set[str]] = defaultdict(set)
        self._norm: dict[str, set[str]] = defaultdict(set)
        for m in data['metrics']:
            for sp in (m['id'], m.get('display_name'), *(m.get('aliases') or [])):
                if sp:
                    self._exact[str(sp).strip().casefold()].add(m['id'])
                    self._norm[_normalize_metric_key(sp)].add(m['id'])
        self.pwc_directions = pwc_directions or {}
        # raw metric name -> set of dataset slugs it was seen on (for the report)
        self.unresolved: dict[str, set[str]] = {}
        # raw metric name -> (why, candidate_ids): 'unknown' (not in registry) vs
        # 'ambiguous_*' (a collision we would not guess through)
        self.unresolved_reason: dict[str, tuple[str, tuple[str, ...]]] = {}
        # metric name -> count of results emitted with an unbounded (inf) bound
        self.unbounded_emitted: dict[str, int] = {}
        # metric name -> set of dataset slugs emitted with a direction that could
        # be resolved from NEITHER the registry NOR the PwC source (an imperfection
        # the strict gate refuses; see _direction). (every_eval_ever#209, Ero #? G2)
        self.direction_unknown: dict[str, set[str]] = {}
        # metric name -> set of dataset slugs where a score could not be reconciled
        # onto the canonical scale (scale_anomaly). Filled by build_results from
        # reconcile_scale's detail; the strict gate refuses these too.
        self.scale_anomalies: dict[str, set[str]] = {}

    def _direction(
        self, registry_dir: bool | None, metric_name: str
    ) -> tuple[bool, str]:
        """Resolve ``lower_is_better`` by a source-priority chain.

        The registry deliberately leaves ``lower_is_better`` ``null`` for metrics
        whose direction is a property of the *use*, not the metric (refusal rate;
        the source's own malformed ``.mean``/``.score`` labels) -- its documented
        intent is "direction stays per-row". So a ``null`` registry direction is
        NOT coerced to ``False`` (which silently asserts higher-is-better); it
        falls back to PwC's own per-metric ``direction`` (the ``metrics`` table),
        and only if THAT is also absent is the direction genuinely unknown. The
        schema requires a bool, so an unknown direction is emitted as ``False``
        (the common default) but always tagged so it is never *silently* wrong --
        the strict gate fails on it; best-effort keeps it flagged.
        Returns ``(lower_is_better, source)`` with source in
        {``registry``, ``pwc_source``, ``unknown``}.
        """
        if registry_dir is not None:
            return bool(registry_dir), 'registry'
        pwc = self.pwc_directions.get(metric_name)
        if pwc == 'lower_is_better':
            return True, 'pwc_source'
        if pwc is not None:  # any other non-null PwC value == higher_is_better
            return False, 'pwc_source'
        return False, 'unknown'

    def _match(self, metric_name: str) -> tuple[str | None, str, tuple[str, ...]]:
        """Return (canonical_id, match_tier, candidate_ids).

        canonical_id is None when the name is unresolvable -- either 'unknown'
        (no candidate) or an 'ambiguous_*' collision (>1 candidate). Exact match
        is tried before the lossy normalized match so distinct-but-similar names
        ('CLIP-IQA' vs 'CLIPIQA+') resolve to their own ids.
        """
        exact = self._exact.get(str(metric_name).strip().casefold())
        if exact:
            if len(exact) == 1:
                return next(iter(exact)), 'exact', ()
            return None, 'ambiguous_exact', tuple(sorted(exact))
        nrm = self._norm.get(_normalize_metric_key(metric_name))
        if nrm:
            if len(nrm) == 1:
                return next(iter(nrm)), 'normalized', ()
            return None, 'ambiguous_normalized', tuple(sorted(nrm))
        return None, 'unknown', ()

    def resolve(
        self,
        metric_name: str,
        obs_range: tuple[float, float],
        dataset_slug: str | None = None,
    ) -> ResolvedMetric:
        obs_min, obs_max = obs_range
        canonical_id, tier, candidates = self._match(metric_name)
        if canonical_id is not None:
            entry = self._by_id[canonical_id]
            score_type = entry.get('score_type') or 'continuous'
            detail: dict[str, Any] = {
                'bound_source': 'registry',
                'canonical_metric_id': entry['id'],
                'match_tier': tier,
                'bound_registry_revision': self.registry_revision,
            }
            # Surface the canonical entry's vetting status/confidence rather than
            # hard-rejecting un-reviewed metrics: EEE should not be held hostage
            # to the registry's review queue, but consumers must be able to see
            # that a bound is still `draft`. (every_eval_ever#209 review, Ero #2)
            if entry.get('review_status'):
                detail['canonical_review_status'] = entry.get('review_status')
            kind, confidence = _metadata_kind_confidence(entry.get('metadata'))
            if kind:
                detail['canonical_metric_kind_flag'] = kind
            if confidence:
                detail['canonical_confidence'] = confidence
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
            lib, dir_source = self._direction(
                entry.get('lower_is_better'), metric_name
            )
            detail['direction_source'] = dir_source
            if dir_source == 'unknown':
                self.direction_unknown.setdefault(metric_name, set())
                if dataset_slug:
                    self.direction_unknown[metric_name].add(dataset_slug)
            return ResolvedMetric(
                metric_id=entry['id'],
                metric_kind=entry['id'],
                lower_is_better=lib,
                score_type=score_type,
                min_score=float(lo),
                max_score=float(hi),
                resolved=True,
                detail=detail,
            )
        # Unresolved: an unknown metric OR an ambiguous collision. Both fail closed
        # by default and are salvageable with --allow-unresolved (observed-range
        # bounds), so a collision is handled by the SAME gate as any un-vetted
        # metric -- no separate flag.
        self.unresolved.setdefault(metric_name, set())
        if dataset_slug:
            self.unresolved[metric_name].add(dataset_slug)
        self.unresolved_reason[metric_name] = (tier, candidates)
        lo, hi = _finite_bounds(min(0.0, obs_min), obs_max)
        lib, dir_source = self._direction(None, metric_name)
        detail = {
            'bound_source': 'observed_unresolved',
            'match_tier': tier,
            'pwc_metric_direction': self.pwc_directions.get(metric_name),
            'direction_source': dir_source,
        }
        # NB: an unresolved metric is already caught by the unresolved gate, so its
        # unknown direction is not *also* tracked in direction_unknown (no double
        # count); the direction_source flag is still recorded for transparency.
        if candidates:
            detail['collision_candidates'] = list(candidates)
        return ResolvedMetric(
            metric_id=f'{SRC}.{snake(metric_name)}',
            metric_kind=snake(metric_name),
            lower_is_better=lib,
            score_type='continuous',
            min_score=lo,
            max_score=hi,
            resolved=False,
            detail=detail,
        )


# PwC's free-text `scale` declarations that map to a real measurement unit. A
# range declaration such as '0-1' or 'unbounded' is NOT a unit -- mapping it into
# metric_unit (as the old `meta.get('scale') or None` did) leaks "0-1"/"unbounded"
# into a field meant for units like proportion/percent/points. Unknown/range-only
# scales leave metric_unit unset; the raw scale is preserved in additional_details
# so nothing is lost (every_eval_ever#209 review, Ero #7).
_SCALE_TO_UNIT = {
    '0-1': 'proportion',
    '0-100': 'percent',
    'percent': 'percent',
    '%': 'percent',
}


def _metric_unit_from_scale(scale: Any) -> str | None:
    if not scale:
        return None
    return _SCALE_TO_UNIT.get(str(scale).strip().lower())


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
        # A unit only when PwC's scale maps to a real one; a range/"unbounded"
        # declaration is preserved raw in additional_details, not forced here.
        metric_unit=_metric_unit_from_scale(meta.get('scale')),
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
                'pwc_scale': meta.get('scale'),
            }
        ),
    )


def score_details(
    ev: dict[str, Any],
    raw_value: Any,
    score: float,
    uncertainty_text: str | None,
    dataset: dict[str, Any],
    paper: dict[str, Any] | None,
    scale_detail: dict[str, Any] | None = None,
) -> ScoreDetails:
    paper = paper or {}
    arxiv_id = paper.get('arxiv_id')
    return ScoreDetails(
        score=score,
        # PwC's '±' spread does not declare itself a standard error, standard
        # deviation, or CI half-width, so we do NOT assert a typed Uncertainty
        # (which would misrepresent it). The reported spread is kept verbatim in
        # `reported_uncertainty` (and within `raw_value`) for downstream
        # interpretation (every_eval_ever#209 review, Ero #5).
        uncertainty=None,
        details=stringify_details(
            {
                **(scale_detail or {}),
                'raw_value': raw_value,
                'reported_uncertainty': uncertainty_text,
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
                'paper_source_url': paper.get('source_url'),
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
    group_medians: dict[tuple[Any, str], float] | None = None,
) -> list[EvaluationResult]:
    """Fan one evaluation row out to one EvaluationResult per jsonb metric."""
    try:
        metrics = json.loads(ev['metrics']) if ev.get('metrics') else {}
    except (TypeError, ValueError):
        metrics = {}
    if not isinstance(metrics, dict) or not metrics:
        return []

    group_medians = group_medians or {}
    ds_slug = dataset.get('slug') or slugify(dataset.get('name'))
    task_slug = (task or {}).get('slug') or 'unknown-task'
    eval_name = f'{SRC}.{snake(task_slug)}.{snake(ds_slug)}'
    src_data = build_source_data(dataset)
    ts = ev.get('evaluated_on') or (str(ev.get('created_at') or '')[:10] or None)

    results: list[EvaluationResult] = []
    for mname, raw in metrics.items():
        score, unc_text = parse_metric_value(raw)
        if score is None:
            continue
        obs_range = metric_ranges.get(mname, (score, score))
        resolved = resolver.resolve(mname, obs_range, ds_slug)
        # The reporting scale is decided once per (dataset, metric) leaderboard
        # from its median, not from this single score (see reconcile_scale).
        group_repr = group_medians.get((ev.get('dataset_id'), mname))
        score, scale_detail = reconcile_scale(
            score,
            resolved.min_score,
            resolved.max_score,
            resolved.resolved,
            group_repr,
        )
        if 'scale_anomaly' in scale_detail:
            resolver.scale_anomalies.setdefault(mname, set()).add(ds_slug)
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
                    ev, raw, score, unc_text, dataset, paper, scale_detail
                ),
            )
        )
    return results


def build_source_metadata(
    dump_version: str,
    source_bucket: str | None = None,
    dump_file: str | None = None,
) -> SourceMetadata:
    # Provenance reflects the ACTUAL source of this run: the HF bucket only when
    # the dump was fetched from one, plus the dump file name. The previous version
    # hardcoded the default bucket even for `--dump` (local) or custom-bucket runs,
    # asserting a provenance that did not hold (every_eval_ever#209 review, mrshu).
    details: dict[str, Any] = {
        'source_role': 'aggregator',
        'dump_version': dump_version,
        'note': (
            'Scores aggregated by Papers with Code from papers and external '
            'leaderboards; not re-run by this adapter.'
        ),
    }
    if source_bucket:
        details['source_bucket'] = source_bucket
    if dump_file:
        details['source_dump_file'] = dump_file
    return SourceMetadata(
        source_name='Papers with Code',
        source_type='documentation',
        source_organization_name='Papers with Code',
        source_organization_url=PWC_SITE,
        # A leaderboard aggregating reported numbers is third_party wrt the model
        # developer, even when a score was self-reported to it (see fields.md).
        evaluator_relationship=EvaluatorRelationship.third_party,
        additional_details=stringify_details(details),
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
    source_bucket: str | None = None,
    dump_file: str | None = None,
    group_medians: dict[tuple[Any, str], float] | None = None,
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
            ev,
            dataset,
            task,
            resolver,
            metric_ranges,
            metric_meta,
            paper,
            group_medians,
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
            source_metadata=build_source_metadata(
                dump_version, source_bucket, dump_file
            ),
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
) -> tuple[
    list[dict[str, Any]],
    dict[str, tuple[float, float]],
    dict[tuple[Any, str], float],
]:
    """Single pass over ``evaluations``. Accumulate, over the WHOLE dump,
    per-metric observed ranges (stable, slice-independent bounds for the
    unresolved fallback) AND, per (dataset, metric) leaderboard within the
    selected slice, the median value used to infer that group's reporting scale
    (see ``reconcile_scale``). The group median ignores ``--limit`` -- the limit
    caps how many rows are EMITTED, not the scale evidence for a leaderboard."""
    ranges: dict[str, list[float]] = defaultdict(
        lambda: [float('inf'), float('-inf')]
    )
    group_values: dict[tuple[Any, str], list[float]] = defaultdict(list)
    selected: list[dict[str, Any]] = []
    for ev in table_rows(dump, 'evaluations'):
        vals = list(_iter_metric_values(ev.get('metrics')))
        for name, val in vals:
            r = ranges[name]
            r[0] = min(r[0], val)
            r[1] = max(r[1], val)
        if dataset_ids is not None and ev.get('dataset_id') not in dataset_ids:
            continue
        for name, val in vals:
            group_values[(ev.get('dataset_id'), name)].append(val)
        if limit is not None and len(selected) >= limit:
            continue
        selected.append(ev)
    metric_ranges = {k: (lo, hi) for k, (lo, hi) in ranges.items()}
    group_medians = {
        key: statistics.median(vals)
        for key, vals in group_values.items()
        if vals
    }
    return selected, metric_ranges, group_medians


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


# The HF *bucket* API (`list_bucket_tree` / `download_bucket_files`) only exists
# in `huggingface_hub>=1.0`, but this repo pins `huggingface-hub>=0.36,<1.0` and
# we deliberately do NOT raise that pin: every EEE consumer would inherit a 1.x
# requirement just so *this one adapter* can auto-download its source. The bucket
# API is therefore treated as an optional, feature-gated capability — the core
# `--dump` path (a dump already on disk) needs only `pgdumplib` and has no such
# requirement. `_require_bucket_api()` imports lazily and fails with an actionable
# message ONLY when auto-download is actually triggered in an environment that
# can't do it. See the adapter README ("Data source") and #209 review (Ero #1).
def _require_bucket_api():
    """Return an ``HfApi``, or exit with a clear remedy if the bucket API is absent.

    The two bucket methods land together in ``huggingface_hub>=1.0``; feature-
    detecting ``list_bucket_tree`` is more robust than parsing a version string
    (and lets the test suite substitute a fake ``HfApi``).
    """
    from huggingface_hub import HfApi

    if not hasattr(HfApi, 'list_bucket_tree'):
        try:
            from importlib.metadata import version

            installed = version('huggingface_hub')
        except Exception:
            installed = 'unknown'
        raise SystemExit(
            'auto-download from the HF bucket needs huggingface_hub>=1.0 for '
            f'the bucket API, but {installed} is installed. Either install a '
            "1.x build here (`pip install 'huggingface_hub>=1.0'`), or pass "
            '--dump <path> to convert a dump already on disk (that path needs '
            'only pgdumplib, no bucket API).'
        )
    return HfApi()


def latest_dump_remote_path(bucket: str, prefix: str = 'postgres') -> str:
    api = _require_bucket_api()
    # The dumps live under `postgres/` in the bucket; list that subtree
    # RECURSIVELY. A non-recursive top-level listing returns the `postgres` dir
    # entry (not the nested `.dump` files) and silently finds nothing.
    dumps = [
        f.path
        for f in api.list_bucket_tree(bucket, prefix=prefix, recursive=True)
        if getattr(f, 'path', '').endswith('.dump')
    ]
    if not dumps:
        raise SystemExit(
            f'no .dump files found under {prefix!r} in bucket {bucket}'
        )
    return sorted(dumps)[-1]


def download_dump(bucket: str, remote_path: str, dest_dir: Path) -> Path:
    api = _require_bucket_api()
    dest_dir.mkdir(parents=True, exist_ok=True)
    local = dest_dir / Path(remote_path).name
    if local.exists():
        print(f'reusing cached dump {local}')
        return local
    print(f'downloading {bucket}:{remote_path} -> {local}')
    api.download_bucket_files(
        bucket, [(remote_path, str(local))], raise_on_missing_files=True
    )
    return local


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def retrieved_ts_from_dump(dump_version: str) -> str:
    """Deterministic Unix-epoch ``retrieved_timestamp`` derived from the dump date.

    ``dump_version`` is the dump's ``YYYYMMDD...`` stamp. Pinning the retrieved
    timestamp to the dump — rather than ``time.time()`` at conversion time — makes
    a re-run over the SAME dump byte-identical, so regenerating the datastore does
    not churn every record's timestamp (every_eval_ever#209 review, Erotemic #3).
    Falls back to the raw version string if it does not start with a parseable
    date. The schema constrains ``retrieved_timestamp`` only to a string
    documented as Unix epoch, so a date-derived epoch is valid.
    """
    try:
        dt = datetime.strptime(str(dump_version)[:8], '%Y%m%d')
    except (ValueError, TypeError):
        return str(dump_version)
    return str(dt.replace(tzinfo=timezone.utc).timestamp())


def replace_output_dir(output_dir: str | Path) -> int:
    """Wipe ``output_dir`` so a re-run REPLACES rather than accumulates records.

    ``save_evaluation_log`` names each file by a fresh ``uuid4``, so re-running
    the adapter into a non-empty dir piles up duplicate records that differ only
    by filename (every_eval_ever#209 review, Erotemic #3). Removing the tree first
    makes the output a pure function of the input dump. Returns the number of
    ``*.json`` records removed (0 if the dir does not exist).
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return 0
    n = sum(1 for _ in output_dir.rglob('*.json'))
    shutil.rmtree(output_dir)
    return n


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
        help='Narrow relaxation: tolerate ONLY unresolved/ambiguous metrics '
        '(emit with observed-range bounds, labelled), while STILL failing on '
        'other imperfections (unknown direction, scale anomaly). Without it the '
        'run fails closed on unresolved metrics so CI never ships un-vetted bounds.',
    )
    ap.add_argument(
        '--best-effort',
        action='store_true',
        help='Emit as much data as possible: every imperfection (unresolved '
        'metric, unknown direction, scale anomaly) is written WITH a flag and the '
        'run exits 0. The default is strict -- ANY imperfection aborts the run '
        'non-zero, giving CI a clean-or-fail signal. Use --best-effort for '
        'exploratory runs or to keep collecting data while fixes are batched; use '
        'the strict default when a run must be perfect (e.g. the commit that fixes '
        'things). Imperfections are always reported regardless of mode.',
    )
    ap.add_argument(
        '--output-dir',
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help=f'Output directory (default: {DEFAULT_OUTPUT_DIR}).',
    )
    return ap.parse_args()


def _report_unresolved(
    unresolved: dict[str, set[str]],
    reasons: dict[str, tuple[str, tuple[str, ...]]] | None = None,
) -> str:
    reasons = reasons or {}
    lines = []
    ambiguous_any = False
    for name, ds in sorted(unresolved.items()):
        why, candidates = reasons.get(name, ('unknown', ()))
        if why.startswith('ambiguous'):
            ambiguous_any = True
            lines.append(
                f'  - {name!r} (on {sorted(ds)}) — AMBIGUOUS: matches '
                f'{list(candidates)}'
            )
        else:
            lines.append(f'  - {name!r} (on {sorted(ds)})')
    msg = (
        f'{len(unresolved)} metric(s) do not resolve in the registry snapshot '
        f'({SNAPSHOT_PATH.name}):\n'
        + '\n'.join(lines)
        + '\n\nAdd them to the eval-card-registry (see its `registry-entity-aliases` '
        'skill) with paper-defined min/max/direction, refresh the snapshot '
        '(refresh_metric_snapshot.py), and re-run — or pass --allow-unresolved to '
        'emit them now with observed-range bounds (labelled bound_source).'
    )
    if ambiguous_any:
        msg += (
            '\n\nAMBIGUOUS metric(s) match more than one canonical id — a '
            'duplicate alias/display_name in the registry, not a missing entry. '
            'Fix the collision in seed/metrics.yaml (remove or uniquify the '
            'offending alias) and refresh the snapshot; --allow-unresolved will '
            'otherwise emit them with observed-range bounds and NO canonical id.'
        )
    return msg


def _summarize_class(title: str, items: dict[str, set[str]]) -> str:
    lines = [f'  - {name!r} (on {sorted(ds)})' for name, ds in sorted(items.items())]
    return f'{len(items)} {title}:\n' + '\n'.join(lines)


def _imperfection_report(resolver: MetricResolver) -> str:
    """Human-readable summary of every imperfection in a run, across all classes,
    printed regardless of run mode (the "noisy" reporting the modes never turn
    off). Empty string when the run was perfect."""
    blocks = []
    if resolver.unresolved:
        blocks.append(
            _report_unresolved(resolver.unresolved, resolver.unresolved_reason)
        )
    if resolver.direction_unknown:
        blocks.append(
            _summarize_class(
                'metric(s) emitted with UNKNOWN direction (no registry direction '
                'and no PwC source direction; lower_is_better defaulted to False, '
                'flagged direction_source=unknown)',
                resolver.direction_unknown,
            )
        )
    if resolver.scale_anomalies:
        blocks.append(
            _summarize_class(
                'metric(s) with a SCALE ANOMALY (a score outside the canonical '
                'range after the group rescale; kept + flagged scale_anomaly, '
                'never rewritten)',
                resolver.scale_anomalies,
            )
        )
    return '\n\n'.join(blocks)


def run(args: argparse.Namespace) -> int:
    if args.dump is not None:
        dump_path = args.dump
        source_bucket: str | None = None  # local dump -> no bucket provenance claim
    else:
        remote = args.remote_path or latest_dump_remote_path(args.bucket)
        dump_path = download_dump(args.bucket, remote, args.raw_dir)
        source_bucket = args.bucket

    dump_version = dump_version_from_path(dump_path)
    dump_file = Path(dump_path).name
    retrieved_ts = retrieved_ts_from_dump(dump_version)

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
        # An EXPLICIT selection that matches nothing is a user error (e.g. a
        # typo'd slug), not an empty-but-successful run: fail loudly rather than
        # writing zero records and exiting 0. A partial match keeps the warning
        # above and proceeds. (every_eval_ever#209 review, mrshu CLI)
        if args.dataset_slugs and not dataset_ids:
            raise SystemExit(
                'ERROR: none of the requested --dataset-slug value(s) matched a '
                f'dataset in the dump: {sorted(slugs)}'
            )

    selected, metric_ranges, group_medians = scan_evaluations(
        dump, dataset_ids, args.limit
    )
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
        source_bucket=source_bucket,
        dump_file=dump_file,
        group_medians=group_medians,
    )

    # --- Imperfection gate -------------------------------------------------
    # Two run modes, one report. The report ("noisy" output) is ALWAYS printed
    # when anything was imperfect, in either mode — modes decide whether to
    # ABORT, never whether to speak. Do this BEFORE wiping the output dir so an
    # aborted run leaves any prior output intact.
    report = _imperfection_report(resolver)
    if report:
        print(report, file=sys.stderr)
    # Which imperfection classes are fatal in strict (default) mode. Unresolved
    # is separately relaxable via --allow-unresolved (the narrow escape hatch);
    # direction_unknown and scale_anomaly are only waived by --best-effort.
    fatal = []
    if resolver.unresolved and not args.allow_unresolved:
        fatal.append(f'{len(resolver.unresolved)} unresolved metric(s)')
    if resolver.direction_unknown:
        fatal.append(
            f'{len(resolver.direction_unknown)} metric(s) with unknown direction'
        )
    if resolver.scale_anomalies:
        fatal.append(
            f'{len(resolver.scale_anomalies)} metric(s) with a scale anomaly'
        )
    if fatal and not args.best_effort:
        raise SystemExit(
            'ERROR: strict mode aborted — ' + '; '.join(fatal) + '. Fix these, '
            'or re-run with --best-effort to emit everything anyway (each '
            'imperfection stays flagged in the output), or --allow-unresolved '
            'to tolerate only the unresolved class. See the report above.'
        )
    if fatal and args.best_effort:
        print(
            'best-effort: emitting despite ' + '; '.join(fatal)
            + ' (all flagged in the output).',
            file=sys.stderr,
        )

    # Replace, don't accumulate: uuid4 filenames mean a re-run would otherwise pile
    # up duplicate records. Wipe only once we know we're going to write.
    removed = replace_output_dir(args.output_dir)
    if removed:
        print(f'replaced output dir: removed {removed} stale record(s)')

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
