"""Tests for the BenchPress aggregator adapter (every_eval_ever/adapters/benchpress/adapter.py)."""
import pytest

from every_eval_ever.adapters.benchpress import adapter
from every_eval_ever.eval_types import EvaluationLog
from every_eval_ever.helpers import SCHEMA_VERSION
from every_eval_ever.helpers.io import SourceRecordsError
from every_eval_ever.validate import validate_file
from every_eval_ever.validator.json_utils import strict_json_loads


def sample_payload() -> dict:
    """In-memory payload (already-parsed lists, as fetch_payload returns).

    Seeds three relationships (a single-provider tech_report -> first_party,
    leaderboard -> third_party, missing source_type -> other), two metric types
    (pct with a declared range; rating, unbounded -> +/-inf), a tech_report
    citation shared by two providers, and one row BenchPress dropped.
    """
    return {
        "metadata": {
            "generated_at_utc": "2026-05-07T04:54:26.048511+00:00",
            "source_git_commit": "5be3b4eddf0188721ff25f00713b589b2cbed8e0",
            "source_data_dirty": False,
            "dataset_revision": "fbe2869d4e1581372830f02a11c64c08365cf656",
        },
        "models": [
            {"id": "gpt-oss-120b", "name": "gpt-oss-120B", "provider": "OpenAI",
             "release_date": "2025-08-05", "open_weights": "true"},
            {"id": "claude-opus-4.6", "name": "Claude Opus 4.6", "provider": "Anthropic",
             "open_weights": "false"},
        ],
        "benchmarks": [
            {"id": "aime_2025", "name": "AIME 2025", "category": "Math",
             "metric": "% correct", "num_problems": 30.0, "source_url": "https://maa.org/aime",
             "canonical_setting": {"metric_type": "pct", "range": [0, 100],
                                   "higher_is_better": True, "version": "AIME-2025-I+II"}},
            {"id": "codeforces_rating", "name": "Codeforces Rating", "category": "Code",
             "metric": "Elo", "source_url": None,
             "canonical_setting": {"metric_type": "rating", "higher_is_better": True}},
        ],
        "scores": [
            {"model_id": "gpt-oss-120b", "benchmark_id": "aime_2025", "score": 97.9,
             "reference_url": "https://arxiv.org/abs/2508.10925", "source_type": "tech_report",
             "audit_status": "verified", "matches_canonical": "true",
             "reported_setting": {"temperature": 0.0, "mode": "thinking", "tools": "none",
                                  "harness": "OLMES", "sampling": "pass@1", "judge": "rule-based"},
             "n_candidates": "1"},
            {"model_id": "gpt-oss-120b", "benchmark_id": "codeforces_rating", "score": 2622.0,
             "reference_url": "https://codeforces.example/x", "source_type": "leaderboard",
             "audit_status": "verified", "reported_setting": {"judge": "gpt-4o"}},
            {"model_id": "claude-opus-4.6", "benchmark_id": "aime_2025", "score": 93.5,
             "reference_url": "https://anthropic.com/news", "source_type": "",
             "audit_status": "verified",
             "reported_setting": {"temperature": 1.0, "mode": "thinking"}},
            # Both providers' scores cite one tech report, so it is a comparison
            # table and neither cell is that report's author reporting itself.
            {"model_id": "claude-opus-4.6", "benchmark_id": "codeforces_rating",
             "score": 2100.0, "reference_url": "https://arxiv.org/abs/2412.19437",
             "source_type": "tech_report", "audit_status": "verified"},
            {"model_id": "gpt-oss-120b", "benchmark_id": "codeforces_rating",
             "score": 2200.0, "reference_url": "https://arxiv.org/abs/2412.19437",
             "source_type": "tech_report", "audit_status": "verified"},
            # BenchPress excludes this from its own canonical matrix.
            {"model_id": "gpt-oss-120b", "benchmark_id": "aime_2025", "score": 12.0,
             "reference_url": "https://example.invalid/rumour",
             "source_type": "official_blog", "audit_status": "dropped"},
        ],
    }


def _logs_by_relationship(developer: str = "openai"):
    """One developer's bundles keyed by relationship — one bundle per split."""
    result = adapter.make_logs(sample_payload())
    return {b.log.source_metadata.evaluator_relationship.value: b
            for b in result.records if b.developer == developer}


def test_relationship_split():
    assert set(_logs_by_relationship()) == {"first_party", "third_party", "other"}
    assert set(_logs_by_relationship("anthropic")) == {"other"}


def test_a_citation_covering_two_providers_is_not_first_party():
    """A tech report's comparison table is not its subjects reporting themselves."""
    others = [
        bundle for bundle in adapter.make_logs(sample_payload()).records
        if bundle.log.source_metadata.evaluator_relationship.value == 'other'
    ]
    cited = {
        result.source_data.url[0]
        for bundle in others
        for result in bundle.log.evaluation_results
    }
    assert "https://arxiv.org/abs/2412.19437" in cited


def test_scores_benchpress_dropped_are_excluded_not_failed():
    """BenchPress rejecting a row is a policy exclusion, not a conversion failure."""
    result = adapter.make_logs(sample_payload())
    assert result.total_records == 6
    assert [e.source_ref for e in result.exclusions] == ['gpt-oss-120b/aime_2025']
    assert 'dropped' in result.exclusions[0].reason
    assert result.failures == []
    result.raise_if_incomplete()  # an exclusion must not fail the run

    kept = adapter.make_logs(sample_payload(), include_unaccepted=True)
    assert kept.exclusions == []


def test_a_score_outside_its_declared_range_is_a_failure(tmp_path):
    """The export mixes scales inside one benchmark; a record cannot state both."""
    payload = sample_payload()
    payload['scores'].append({
        'model_id': 'gpt-oss-120b', 'benchmark_id': 'aime_2025', 'score': 950.0,
        'reference_url': 'https://arxiv.org/abs/2508.10925',
        'source_type': 'tech_report', 'audit_status': 'verified',
    })
    result = adapter.make_logs(payload)
    assert [f.source_ref for f in result.failures] == ['gpt-oss-120b/aime_2025']
    assert 'declared range [0.0, 100.0]' in result.failures[0].reason
    with pytest.raises(SourceRecordsError):
        result.raise_if_incomplete()

    # The valid records still publish, and nothing invalid reaches the tree.
    for path in adapter.export_logs(result.records, tmp_path / 'data' / 'benchpress'):
        assert validate_file(path).valid


def test_logs_are_schema_valid():
    for bundle in adapter.make_logs(sample_payload()).records:
        validated = EvaluationLog.model_validate(bundle.log.model_dump())
        assert validated.schema_version == SCHEMA_VERSION
        assert validated.source_metadata.source_type.value == "documentation"
        assert validated.source_metadata.source_organization_name == "BenchPress"
        assert validated.eval_library.name == "BenchPress"


def test_model_id_and_evaluation_id():
    fp = _logs_by_relationship()["first_party"].log
    assert fp.model_info.id == "openai/gpt-oss-120b"
    assert fp.model_info.additional_details["benchpress_model_id"] == "gpt-oss-120b"
    # retrieved_timestamp derives from metadata.generated_at_utc
    assert fp.evaluation_id.startswith("benchpress/first_party/openai_gpt-oss-120b/")
    assert fp.retrieved_timestamp == adapter._iso_to_epoch_str(
        "2026-05-07T04:54:26.048511+00:00")


def test_citation_url_and_reported_by():
    res = _logs_by_relationship()["first_party"].log.evaluation_results[0]
    assert res.source_data.url[0] == "https://arxiv.org/abs/2508.10925"
    assert res.source_data.additional_details["reported_by"] == "arxiv.org"
    assert res.source_data.additional_details["source_role"] == "aggregator"


def test_bounded_metric_uses_declared_range():
    pct = _logs_by_relationship()["first_party"].log.evaluation_results[0].metric_config
    assert pct.score_type.value == "continuous"
    assert (pct.min_score, pct.max_score) == (0.0, 100.0)


def test_unbounded_metric_uses_infinity():
    rating = _logs_by_relationship()["third_party"].log.evaluation_results[0].metric_config
    assert rating.metric_kind == "rating"
    assert rating.min_score == float("-inf")
    assert rating.max_score == float("inf")


def test_version_provenance_recorded():
    details = _logs_by_relationship()["first_party"].log.source_metadata.additional_details
    assert details["benchpress_source_git_commit"] == "5be3b4eddf0188721ff25f00713b589b2cbed8e0"
    assert details["benchpress_generated_at_utc"] == "2026-05-07T04:54:26.048511+00:00"
    assert details["benchpress_dataset_revision"] == (
        "fbe2869d4e1581372830f02a11c64c08365cf656")


def test_export_writes_standards_compliant_infinity_and_validates(tmp_path):
    paths = adapter.export_logs(adapter.make_logs(sample_payload()).records, tmp_path)
    assert len(paths) == 4
    inf_raws = [p.read_text() for p in paths if "Infinity" in p.read_text()]
    assert inf_raws
    # Unbounded bounds are the JSON *string* "Infinity", which a strict parser
    # accepts (a bare Infinity token would fail here) and pydantic reads as a float.
    assert '"Infinity"' in inf_raws[0]
    reloaded = EvaluationLog.model_validate(strict_json_loads(inf_raws[0]))
    assert any(result.metric_config.max_score == float("inf")
               for result in reloaded.evaluation_results)
    for p in paths:
        report = validate_file(p)
        assert report.valid, report.errors
        assert p.parent.parent.parent == tmp_path  # <out>/<dev>/<model>/<uuid>.json
    assert (tmp_path / "openai" / "gpt-oss-120b").is_dir()
    assert (tmp_path / "anthropic" / "claude-opus-4.6").is_dir()
