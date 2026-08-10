"""What a metric's name tells us about its range and its polarity.

Upstream harnesses report a metric's value and almost never its range, so the
range has to come from the metric's definition. The names below mean the same
thing in every harness we convert, so their bounds are shared; anything whose
scale is harness-specific belongs in that converter's own table, layered on top
of `SHARED_METRIC_BOUNDS`. `bleu` is the cautionary example: sacrebleu (lm-eval)
reports 0-100 while nltk's `sentence_bleu` (HELM's `bleu_1`/`bleu_4`) reports
0-1, so the bare name cannot carry a range.

A metric that is in no table gets no bounds at all and a `bounds_status` marker,
because `min_score`/`max_score` are nullable and "not provided" is true, while
[0, 1] on an unbounded metric is not.
"""

from __future__ import annotations

from every_eval_ever.eval_types import ScoreType

# Infinite bounds are serialized as the JSON strings "Infinity"/"-Infinity".
SHARED_METRIC_BOUNDS: dict[str, tuple[float, float]] = {
    'accuracy': (0.0, 1.0),
    'acc': (0.0, 1.0),
    'acc_norm': (0.0, 1.0),
    'em': (0.0, 1.0),
    'exact_match': (0.0, 1.0),
    'f1': (0.0, 1.0),
    'f1_score': (0.0, 1.0),
    'precision': (0.0, 1.0),
    'recall': (0.0, 1.0),
    'mcc': (-1.0, 1.0),
    'brier_score': (0.0, 1.0),
    # Dispersion of a score distribution, not a score: non-negative, and
    # unbounded above unless the underlying metric is bounded.
    'std': (0.0, float('inf')),
    'stddev': (0.0, float('inf')),
    'stderr': (0.0, float('inf')),
    'bootstrap_stderr': (0.0, float('inf')),
    'var': (0.0, float('inf')),
}

# Metrics whose definition fixes the direction, so no harness has to say so.
LOWER_IS_BETTER: frozenset[str] = frozenset(
    {
        'bits_per_byte',
        'brier_score',
        'byte_perplexity',
        'calibration_error',
        'cer',
        'ece',
        'perplexity',
        'ter',
        'wer',
        'word_perplexity',
    }
)

# Metrics that summarize the spread of a score distribution. "Better" does not
# apply to them, and the schema has fields for them on the score they describe
# (`uncertainty.standard_error`, `uncertainty.standard_deviation`), so a
# converter should prefer routing them there over emitting them as scores.
DISPERSION_METRICS: frozenset[str] = frozenset(
    {'std', 'stddev', 'stderr', 'bootstrap_stderr', 'var'}
)

_UNKNOWN_BOUNDS = {'bounds_status': 'unknown'}
_NO_POLARITY = {'polarity': 'not_applicable'}


def metric_bounds_fields(
    metric_name: str | None,
    bounds_table: dict[str, tuple[float, float]] | None = None,
) -> dict[str, object]:
    """The `MetricConfig` fields that describe one metric's range and direction.

    Spread into a `MetricConfig(...)` call. `bounds_table` overrides and extends
    the shared bounds for harnesses that spell a metric differently or report it
    on a different scale.
    """
    table = (
        {**SHARED_METRIC_BOUNDS, **bounds_table}
        if bounds_table
        else SHARED_METRIC_BOUNDS
    )
    name = metric_name or ''
    bounds = table.get(name)
    details = dict(_NO_POLARITY) if name in DISPERSION_METRICS else {}

    if bounds is None:
        return {
            'lower_is_better': name in LOWER_IS_BETTER,
            'additional_details': {**details, **_UNKNOWN_BOUNDS},
        }
    return {
        'lower_is_better': name in LOWER_IS_BETTER,
        'score_type': ScoreType.continuous,
        'min_score': bounds[0],
        'max_score': bounds[1],
        'additional_details': details or None,
    }


def count_unknown_bounds(metric_configs) -> int:
    """How many of these metric configs have no known range."""
    return sum(
        config.additional_details is not None
        and config.additional_details.get('bounds_status') == 'unknown'
        for config in metric_configs
    )
