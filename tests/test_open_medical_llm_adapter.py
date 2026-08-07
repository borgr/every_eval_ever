"""Tests for the Open Medical-LLM Leaderboard adapter. Offline — builds records
from synthetic lm-evaluation-harness result objects (no network)."""
import json
import pathlib

import pytest

from every_eval_ever.adapters.open_medical_llm import adapter
from every_eval_ever.eval_types import EvaluationLog
from every_eval_ever.helpers import SCHEMA_VERSION
from every_eval_ever.helpers.io import SourceRecordsError


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
    assert v.schema_version == SCHEMA_VERSION
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
    # accuracy is a global registry metric: one join key across benchmarks, kept
    # apart by evaluation_name rather than by a per-benchmark metric id
    assert {r.metric_config.metric_id for r in v.evaluation_results} == {"accuracy"}
    # source_data points at the benchmark's OWN dataset repo, not the results repo
    med = next(r for r in v.evaluation_results if r.evaluation_name.endswith(".medmcqa"))
    assert med.source_data.hf_repo == "openlifescienceai/medmcqa"
    # stderr -> uncertainty when present; absent otherwise
    assert med.score_details.uncertainty.standard_error.value == 0.012
    anat = next(r for r in v.evaluation_results if r.evaluation_name.endswith(".mmlu_anatomy"))
    assert anat.score_details.uncertainty is None
    # evaluation_id keyed on the eval time (stable), not `now`; sub-second precision
    # is preserved so same-second runs stay distinct (filename fraction .123 -> .123000)
    assert v.evaluation_id == "open-medical-llm-leaderboard/acme_med-x/1714521600.123000"
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


def test_resolution_decoupled_from_evaluation_id():
    # A registry-canonical id that DIFFERS from the source path must become
    # model_info.id (the join key) WITHOUT moving evaluation_id (keyed on the raw
    # repo, so re-ingest stays idempotent even if the draft is later re-mapped).
    built = adapter.make_log(
        "acme/med-x", _results_obj(),
        "acme/med-x/results_2024-05-01 00:00:00.123.json", "1700000000.0",
        model_id="acme/OpenBioLLM-med-x",
        resolution_details={"model_id_resolution": "registry",
                            "model_id_resolution_strategy": "fuzzy_stem",
                            "model_id_resolution_confidence": 0.72,
                            "model_id_created_new": True,
                            "model_id_review_status": "draft"},
    )
    log = built[0]
    assert log.model_info.id == "acme/OpenBioLLM-med-x"        # resolved canonical -> join key
    assert log.evaluation_id == "open-medical-llm-leaderboard/acme_med-x/1714521600.123000"  # RAW slug
    ad = log.model_info.additional_details
    assert ad["source_model_repo"] == "acme/med-x"            # raw->canonical mapping recorded
    assert ad["model_id_created_new"] == "true"               # provenance stringified
    assert ad["model_id_review_status"] == "draft"


def test_resolve_model_id_offline_no_network():
    # opt-out path must not touch the network and must fall back to the path id
    mid, prov = adapter.resolve_model_id("acme/med-x", enabled=False)
    assert mid == "acme/med-x"
    assert prov == {"model_id_resolution": "offline"}


def test_needs_registry_review_flags_unverified():
    assert adapter._needs_registry_review({"model_id_resolution": "registry",
                                           "model_id_review_status": "reviewed",
                                           "model_id_resolution_confidence": 1.0}) is False
    assert adapter._needs_registry_review({"model_id_created_new": True}) is True
    assert adapter._needs_registry_review({"model_id_resolution": "unreachable"}) is True
    assert adapter._needs_registry_review({"model_id_review_status": "draft"}) is True
    assert adapter._needs_registry_review({"model_id_resolution_confidence": 0.5}) is True


def test_fractional_timestamp_distinguishes_same_second():
    # two runs of the same model in the same whole second -> distinct evaluation_id
    a = adapter.make_log("acme/med-x", _results_obj(),
                         "acme/med-x/results_2024-05-01 00:00:00.111.json", "1700000000.0")[0]
    b = adapter.make_log("acme/med-x", _results_obj(),
                         "acme/med-x/results_2024-05-01 00:00:00.222.json", "1700000000.0")[0]
    assert a.evaluation_id != b.evaluation_id


def test_model_metadata_axes_follow_the_run_config():
    """A locally loaded checkpoint evidences both axes; without it, no claim."""
    log = adapter.make_log("acme/med-x", _results_obj(),
                           "acme/med-x/results_2024-05-01 00:00:00.json", "1700000000.0")[0]
    ad = log.model_info.additional_details
    assert ad["deployment_type"] == "self_deployed"
    assert ad["model_availability"] == "open_weights"

    obj = _results_obj()
    obj["config"]["model_args"] = ""            # older rows record no checkpoint
    ad = adapter.make_log("acme/med-x", obj,
                          "acme/med-x/results_2024-05-01 00:00:00.json",
                          "1700000000.0")[0].model_info.additional_details
    assert ad["deployment_type"] == "unknown"
    assert ad["model_availability"] == "unknown"


def test_config_model_repo_prefers_pretrained():
    assert adapter.config_model_repo(
        {"model_args": "pretrained=acme/med-x,revision=main", "model_name": "other/x"}
    ) == "acme/med-x"
    assert adapter.config_model_repo(
        {"model_args": {"pretrained": "acme/med-x"}}) == "acme/med-x"
    assert adapter.config_model_repo({"model_name": "acme/med-x"}) == "acme/med-x"
    assert adapter.config_model_repo({}) is None


def test_agreeing_path_and_config_use_the_path():
    repo, prov = adapter.evaluated_model_repo("acme/med-x", _results_obj()["config"])
    assert repo == "acme/med-x"
    assert prov == {"model_identity_source": "dataset_path"}


def test_divergent_identity_is_reconciled_through_hf_aliases(monkeypatch):
    """Two spellings of one repo are the same model; two repos are not resolvable."""
    config = {"model_args": "pretrained=acme/MedX"}
    monkeypatch.setattr(adapter, "canonical_hf_repo", lambda repo, **kw: "acme/Med-X")
    repo, prov = adapter.evaluated_model_repo("acme/med-x", config)
    assert repo == "acme/Med-X"                       # the id both aliases point at
    assert prov["model_identity_source"] == "hf_alias"
    assert prov["model_identity_run_config"] == "acme/MedX"

    monkeypatch.setattr(adapter, "canonical_hf_repo", lambda repo, **kw: repo)
    repo, prov = adapter.evaluated_model_repo("acme/med-x", config)
    assert repo is None                               # genuinely different models
    assert prov["model_identity_source"] == "conflicting_repos"


def test_divergent_identity_is_not_guessed_offline():
    repo, prov = adapter.evaluated_model_repo(
        "acme/med-x", {"model_args": "pretrained=acme/MedX"}, check_aliases=False
    )
    assert repo is None
    assert prov["model_identity_source"] == "unresolved_offline"


def test_every_selected_file_is_accounted_for(monkeypatch):
    """A selected file that yields no record is a failure, not a silent skip."""
    chosen = {
        "acme/med-x": "acme/med-x/results_2024-05-01 00:00:00.json",
        "acme/empty": "acme/empty/results_2024-05-01 00:00:00.json",
    }
    objs = {
        chosen["acme/med-x"]: _results_obj(),
        chosen["acme/empty"]: {"config": {"model_name": "acme/empty"},
                               "results": {"pubmedqa": {"f1,none": 0.4}}},
    }
    monkeypatch.setattr(adapter, "fetch_json", lambda path: objs[path])
    result, flagged = adapter.convert(
        chosen, ["GPT-4/results_2024-03-01T00:00:00.json"], "1700000000.0",
        resolve_enabled=False, workers=1,
    )
    assert flagged == []
    assert result.total_records == 3
    assert len(result.records) == 1
    assert [f.source_ref for f in result.failures] == [chosen["acme/empty"]]
    assert "acc,none" in result.failures[0].reason
    # the hand-curated baseline is a documented exclusion, not a failure
    assert [e.source_ref for e in result.exclusions] == [
        "GPT-4/results_2024-03-01T00:00:00.json"]
    with pytest.raises(SourceRecordsError):
        result.raise_if_incomplete()


def test_existing_records_are_reported_before_a_second_copy_is_written(tmp_path):
    """Fresh uuid filenames mean a re-run would duplicate, not replace."""
    target = tmp_path / "acme" / "med-x"
    target.mkdir(parents=True)
    (target / "0d2f9f1e-0000-4000-8000-000000000000.json").write_text("{}")
    assert adapter.existing_records(str(tmp_path), [("acme", "med-x")]) == [
        target / "0d2f9f1e-0000-4000-8000-000000000000.json"]
    assert adapter.existing_records(str(tmp_path), [("acme", "other")]) == []


def test_failure_report_survives_a_publication_error(tmp_path, monkeypatch):
    """The report accounts for the conversion, so publication must not take it down."""
    paths = ["acme/med-x/results_2024-05-01 00:00:00.json",
             "acme/empty/results_2024-05-01 00:00:00.json"]
    objs = {paths[0]: _results_obj(),
            paths[1]: {"config": {"model_name": "acme/empty"},
                       "results": {"pubmedqa": {"f1,none": 0.4}}}}
    monkeypatch.setattr(adapter, "list_result_files", lambda: paths)
    monkeypatch.setattr(adapter, "fetch_json", lambda path: objs[path])
    monkeypatch.setattr(adapter, "save_evaluation_logs",
                        lambda outputs: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr("sys.argv", ["adapter", "--output-dir", str(tmp_path / "data"),
                                     "--no-registry-resolve", "--workers", "1"])
    with pytest.raises(RuntimeError):
        adapter.main()
    report = json.loads(pathlib.Path(
        adapter.default_failure_report_path(str(tmp_path / "data"))).read_text())
    assert [f["source_ref"] for f in report["failed_records"]] == [paths[1]]


def test_next_link_parses_rel_next():
    h = ('<https://huggingface.co/api/datasets/x/tree/main?recursive=true&cursor=ABC>; rel="next", '
         '<https://huggingface.co/api/datasets/x/tree/main>; rel="first"')
    assert adapter._next_link(h).endswith("cursor=ABC")
    assert adapter._next_link(None) is None
    assert adapter._next_link('<https://x>; rel="prev"') is None
