"""HELM metric filtering helpers."""

from __future__ import annotations

from typing import Optional

# HELM emits both benchmark metrics and bookkeeping telemetry in stats.json /
# per_instance_stats.json. In this PR, only benchmark-quality metrics become
# EEE aggregate/detail metric rows. Bookkeeping can be mapped to token_usage,
# performance, metadata, or additional_details in a future follow-up.
CORE_METRIC_PREFIXES: tuple[str, ...] = (
    'exact_match',
    'quasi_exact_match',
    'prefix_exact_match',
    'quasi_prefix_exact_match',
    'classification_micro_f1',
    'classification_macro_f1',
    'f1_score',
    'rouge_l',
    'bleu_',
    'ifeval_strict_accuracy',
    'chain_of_thought_correctness',
    'math_equiv',
    'math_equiv_chain_of_thought',
)


def is_core_metric(metric_name: Optional[str]) -> bool:
    """Return True when a HELM stat should become an EEE metric row."""
    return bool(metric_name) and any(
        metric_name.startswith(prefix) for prefix in CORE_METRIC_PREFIXES
    )


# Bounds for the metrics HELM spells or scales its own way, layered over
# converters/common/metrics.py::SHARED_METRIC_BOUNDS. HELM's `bleu_1`/`bleu_4`
# are nltk's `sentence_bleu` and its `f1_score` is nltk's `f_measure`, so both
# are 0-1 here, unlike lm-eval's `bleu`, which is sacrebleu's 0-100.
HELM_METRIC_BOUNDS: dict[str, tuple[float, float]] = {
    'quasi_exact_match': (0.0, 1.0),
    'prefix_exact_match': (0.0, 1.0),
    'quasi_prefix_exact_match': (0.0, 1.0),
    'classification_micro_f1': (0.0, 1.0),
    'classification_macro_f1': (0.0, 1.0),
    'rouge_l': (0.0, 1.0),
    'bleu_1': (0.0, 1.0),
    'bleu_4': (0.0, 1.0),
    'ifeval_strict_accuracy': (0.0, 1.0),
    'chain_of_thought_correctness': (0.0, 1.0),
    'math_equiv': (0.0, 1.0),
    'math_equiv_chain_of_thought': (0.0, 1.0),
}


def metric_bounds_name(metric_name: str) -> str:
    """The name to look bounds up under, without HELM's ``@k`` suffix.

    ``exact_match@5`` is ``exact_match`` over the best of five completions: the
    suffix says how many completions were considered, not on what scale the
    result is reported.
    """
    return metric_name.split('@', 1)[0]
