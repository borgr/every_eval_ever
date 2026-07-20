"""Tests for the Open Medical-LLM Leaderboard adapter. Offline — builds records
from synthetic lm-evaluation-harness result objects (no network)."""
from every_eval_ever.eval_types import EvaluationLog
from utils.open_medical_llm import adapter


def _results_obj():
    return {
        "config": {"model_name": "acme/med-x", "model_dtype": "float16",
                   "model_args": "pretrained=acme/med-x"},
        "results": {
            "medmcqa": {"acc,none": 0.61, "acc_stderr,none": 0.012, "alias": "medmcqa"},
            "pubmedqa": {"acc,none": 0.75, "acc_stderr,none": 0.02},
            "mmlu_anatomy": {"acc,none": 0.55},  # no stderr -> no uncertainty
        },
    }


def test_make_log_is_valid_and_mapped():
    # real result filenames are `results_<YYYY-MM-DD HH:MM:SS.micros>.json`
    built = adapter.make_log("acme/med-x", _results_obj(),
                             "acme/med-x/results_2024-05-01 00:00:00.123.json", "1700000000.0")
    assert built is not None
    log, developer, model = built
    v = EvaluationLog.model_validate(log.model_dump())  # schema-valid
    assert v.schema_version == "0.2.2"
    assert developer == "acme" and model == "med-x"
    assert v.model_info.id == "acme/med-x"                 # HF-form id as-is
    assert v.source_metadata.source_type.value == "documentation"
    assert v.source_metadata.evaluator_relationship.value == "third_party"
    # documentation source WITH a known harness (format is unmistakable)
    assert v.eval_library.name == "lm-evaluation-harness"
    # one result per medical benchmark; accuracy / proportion / [0,1]
    names = {r.evaluation_name for r in v.evaluation_results}
    assert names == {"open-medical-llm-leaderboard.medmcqa",
                     "open-medical-llm-leaderboard.pubmedqa",
                     "open-medical-llm-leaderboard.mmlu_anatomy"}
    r0 = v.evaluation_results[0].metric_config
    assert (r0.score_type.value, r0.min_score, r0.max_score) == ("continuous", 0.0, 1.0)
    assert r0.metric_kind == "accuracy"
    # source_data points at the benchmark's OWN dataset repo, not the results repo
    med = next(r for r in v.evaluation_results if r.evaluation_name.endswith(".medmcqa"))
    assert med.source_data.hf_repo == "openlifescienceai/medmcqa"
    # stderr -> uncertainty when present; absent otherwise
    assert med.score_details.uncertainty.standard_error.value == 0.012
    anat = next(r for r in v.evaluation_results if r.evaluation_name.endswith(".mmlu_anatomy"))
    assert anat.score_details.uncertainty is None
    # evaluation_id keyed on the eval time (stable), not `now`
    assert v.evaluation_id == "open-medical-llm-leaderboard/acme_med-x/1714521600"
    assert v.retrieved_timestamp == "1700000000.0"


def test_evaluation_id_stable_when_timestamp_unparseable():
    # a filename with no parseable timestamp must NOT key evaluation_id on `now`
    # (retrieved_ts) — it must be stable across re-runs so re-ingest is idempotent.
    path = "acme/med-x/results_no_timestamp_here.json"
    a = adapter.make_log("acme/med-x", _results_obj(), path, "1700000000.0")[0]
    b = adapter.make_log("acme/med-x", _results_obj(), path, "1800000000.0")[0]  # later run
    assert a.evaluation_id == b.evaluation_id                    # idempotent
    assert "1700000000" not in a.evaluation_id                   # not keyed on `now`
    assert a.evaluation_timestamp is None                        # run time truly unknown


def test_make_result_needs_accuracy():
    assert adapter.make_result("pubmedqa", {"f1,none": 0.5}, None) is None  # no acc,none


def test_latest_per_model_skips_baselines_and_picks_latest():
    paths = [
        "acme/med-x/results_2024-01-01T00:00:00.json",
        "acme/med-x/results_2024-06-01T00:00:00.json",   # latest
        "GPT-4/results_2024-03-01T00:00:00.json",         # 2-seg baseline -> skipped
    ]
    chosen, baselines = adapter.latest_per_model(paths)
    assert chosen == {"acme/med-x": "acme/med-x/results_2024-06-01T00:00:00.json"}
    assert baselines == ["GPT-4/results_2024-03-01T00:00:00.json"]
