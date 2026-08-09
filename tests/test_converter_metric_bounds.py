"""What `converters/common/metrics.py` promises to every converter that uses it.

The three harness converters each have their own tests for the records they
produce; these pin the shared rules those records rely on, so a change here shows
up as one failure with a name rather than as three unrelated ones.
"""

from every_eval_ever.converters.common.metrics import (
    count_unknown_bounds,
    metric_bounds_fields,
)
from every_eval_ever.converters.helm.metrics import (
    HELM_METRIC_BOUNDS,
    metric_bounds_name,
)
from every_eval_ever.converters.lm_eval.utils import LM_EVAL_METRIC_BOUNDS
from every_eval_ever.eval_types import MetricConfig, ScoreType


def _config(metric_name: str, bounds_table=None) -> MetricConfig:
    """Build the MetricConfig a converter would build for this metric."""
    return MetricConfig(
        evaluation_description=metric_name,
        metric_name=metric_name,
        **metric_bounds_fields(metric_name, bounds_table),
    )


def test_a_known_metric_gets_its_range_and_its_direction():
    config = _config('accuracy')

    assert (config.min_score, config.max_score) == (0.0, 1.0)
    assert config.score_type == ScoreType.continuous
    assert config.lower_is_better is False
    assert config.additional_details is None


def test_an_unknown_metric_gets_no_range_at_all():
    """`min_score`/`max_score` are nullable, so "not provided" is available and true.

    [0, 1] on a metric whose scale we have not checked is not.
    """
    config = _config('semantic_similarity_v3')

    assert config.min_score is None
    assert config.max_score is None
    assert config.additional_details == {'bounds_status': 'unknown'}


def test_a_metric_whose_definition_fixes_its_direction_says_so():
    assert _config('word_perplexity').lower_is_better is True
    assert _config('exact_match').lower_is_better is False


def test_a_dispersion_metric_claims_no_direction():
    """`lower_is_better` is required, so `False` is what an inapplicable direction
    serializes as, and the marker is what carries the caveat."""
    config = _config('std')

    assert (config.min_score, config.max_score) == (0.0, float('inf'))
    assert config.additional_details == {'polarity': 'not_applicable'}


def test_the_same_metric_name_can_mean_two_scales():
    """lm-eval's `bleu` is sacrebleu's 0-100; HELM's `bleu_1` is nltk's 0-1.

    This is why the bounds are layered per harness rather than kept in one table
    keyed by a bare metric name.
    """
    lm_eval_bleu = _config('bleu', LM_EVAL_METRIC_BOUNDS)
    helm_bleu = _config('bleu_1', HELM_METRIC_BOUNDS)

    assert (lm_eval_bleu.min_score, lm_eval_bleu.max_score) == (0.0, 100.0)
    assert (helm_bleu.min_score, helm_bleu.max_score) == (0.0, 1.0)


def test_a_harness_table_can_override_the_shared_bounds():
    shared = _config('accuracy')
    overridden = _config('accuracy', {'accuracy': (0.0, 100.0)})

    assert shared.max_score == 1.0
    assert overridden.max_score == 100.0


def test_helm_at_k_suffix_does_not_cost_a_metric_its_bounds():
    assert metric_bounds_name('exact_match@5') == 'exact_match'
    assert metric_bounds_name('exact_match') == 'exact_match'

    config = _config(
        metric_bounds_name('quasi_exact_match@5'), HELM_METRIC_BOUNDS
    )
    assert (config.min_score, config.max_score) == (0.0, 1.0)


def test_unknown_bounds_are_countable_for_the_record_they_end_up_in():
    configs = [_config('accuracy'), _config('vibes'), _config('more_vibes')]

    assert count_unknown_bounds(configs) == 2
