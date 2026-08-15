from __future__ import annotations

import json
from pathlib import Path

import pytest

from every_eval_ever.adapters.llm_stats import adapter
from every_eval_ever.eval_types import EvaluationLog
from every_eval_ever.helpers.io import SourceRecordsError
from every_eval_ever.validate import validate_file


def sample_payload() -> dict:
    return {
        'models': {
            'data': [
                {
                    'id': 'gpt-5',
                    'slug': 'gpt-5',
                    'name': 'GPT-5',
                    'provider': {'slug': 'openai', 'name': 'OpenAI'},
                    'context_window': 128000,
                    'modalities': ['text'],
                    'pricing': {'input': 1.25, 'output': 10.0},
                    'release_date': '2025-08-07',
                    'license': 'proprietary',
                },
                {
                    'id': 'claude-4-opus',
                    'slug': 'claude-4-opus',
                    'name': 'Claude 4 Opus',
                    'provider': {'slug': 'anthropic', 'name': 'Anthropic'},
                    'context_window': 200000,
                    'modalities': ['text', 'vision'],
                },
            ]
        },
        'benchmarks': {
            'data': [
                {
                    'id': 'gpqa-diamond',
                    'slug': 'gpqa-diamond',
                    'name': 'GPQA Diamond',
                    'description': 'Graduate-level science QA.',
                    'category': 'reasoning',
                    'min_score': 0,
                    'max_score': 100,
                    'metric_kind': 'accuracy',
                    'metric_unit': 'percent',
                },
                {
                    'id': 'math-500',
                    'slug': 'math-500',
                    'name': 'MATH-500',
                    'description': 'Competition mathematics benchmark.',
                    'category': 'math',
                    'min_score': 0,
                    'max_score': 1,
                    'metric_kind': 'accuracy',
                    'metric_unit': 'proportion',
                },
            ]
        },
        'scores': {
            'data': [
                {
                    'id': 'score-gpt5-gpqa',
                    'model_id': 'gpt-5',
                    'benchmark_id': 'gpqa-diamond',
                    'score': 94.2,
                    'unit': 'percent',
                    'source_type': 'model_card',
                    'verified': True,
                    'source_url': 'https://openai.com/index/gpt-5-system-card/',
                },
                {
                    'id': 'score-gpt5-math',
                    'model_id': 'gpt-5',
                    'benchmark_id': 'math-500',
                    'score': 0.91,
                    'unit': 'proportion',
                    'provenance': 'independent_runner',
                    'verification_tier': 'third_party',
                    'citation_url': 'https://example.org/independent-math-500',
                },
                {
                    'id': 'score-claude-gpqa',
                    'model_id': 'claude-4-opus',
                    'benchmark_id': 'gpqa-diamond',
                    'score': 88.5,
                },
            ]
        },
    }


def logs_by_relationship() -> dict[str, EvaluationLog]:
    bundles = adapter.make_logs(
        sample_payload(),
        base_url=adapter.DEFAULT_BASE_URL,
        retrieved_timestamp='1234567890.0',
    )
    logs = {
        bundle.log.source_metadata.evaluator_relationship.value: bundle.log
        for bundle in bundles
    }
    assert set(logs) == {'first_party', 'third_party', 'other'}
    return logs


def test_make_logs_validate_against_schema():
    for log in logs_by_relationship().values():
        validated = EvaluationLog.model_validate(log.model_dump())
        assert validated.source_metadata.source_organization_name == 'LLM Stats'
        assert validated.source_metadata.source_type.value == 'documentation'
        assert (
            validated.source_metadata.additional_details['attribution_required']
            == 'true'
        )
        assert (
            validated.source_metadata.additional_details['scores_endpoint']
            == 'https://api.llm-stats.com/v1/scores'
        )
        assert (
            validated.source_metadata.additional_details[
                'scores_endpoint_fallback'
            ]
            == 'https://api.llm-stats.com/leaderboard/benchmarks/{benchmark_id}'
        )


def test_scores_are_grouped_by_evaluator_relationship():
    logs = logs_by_relationship()

    first_party = logs['first_party']
    assert first_party.model_info.id == 'openai/gpt-5'
    assert first_party.evaluation_id.startswith(
        'llm-stats/first_party/openai_gpt-5/'
    )
    assert len(first_party.evaluation_results) == 1
    assert (
        first_party.evaluation_results[0].evaluation_name
        == 'llm_stats.gpqa-diamond'
    )

    third_party = logs['third_party']
    assert third_party.model_info.id == 'openai/gpt-5'
    assert len(third_party.evaluation_results) == 1
    assert (
        third_party.evaluation_results[0].evaluation_name
        == 'llm_stats.math-500'
    )

    other = logs['other']
    assert other.model_info.id == 'anthropic/claude-4-opus'
    assert other.source_metadata.evaluator_relationship.value == 'other'


def test_raw_citation_and_provenance_are_preserved():
    logs = logs_by_relationship()

    first_result = logs['first_party'].evaluation_results[0]
    first_details = first_result.score_details.details or {}
    assert first_details['raw_provenance_label'] == 'model_card'
    assert first_details['raw_verified'] == 'true'
    assert first_details['raw_source_organization'] == 'openai'
    assert first_details['relationship_inference_reason'] == (
        'source_matches_model_developer'
    )
    assert 'https://openai.com/index/gpt-5-system-card/' in json.loads(
        first_details['source_urls_json']
    )
    assert (
        'https://openai.com/index/gpt-5-system-card/'
        in first_result.source_data.url
    )
    assert first_result.metric_config.metric_unit == 'percent'
    assert first_result.metric_config.max_score == 100

    other_result = logs['other'].evaluation_results[0]
    other_details = other_result.score_details.details or {}
    assert other_details['raw_provenance_label'] == 'unknown'
    assert other_details['relationship_inference_reason'] == (
        'no_provenance_signal'
    )


def test_fallback_fetch_failures_are_preserved_and_fail_conversion(
    monkeypatch,
):
    payload = sample_payload()

    def fake_fetch(url, *, headers):
        del headers
        if url.endswith('/v1/models'):
            return payload['models']
        if url.endswith('/leaderboard/benchmarks'):
            return payload['benchmarks']
        if url.endswith('/v1/scores'):
            raise adapter.FetchError('scores endpoint unavailable')
        raise adapter.FetchError(f'benchmark detail unavailable: {url}')

    monkeypatch.setattr(adapter, 'fetch_json', fake_fetch)

    fetched = adapter.fetch_payload('secret', 'https://example.test')

    assert len(fetched['source_failures']) == 2
    assert (
        fetched['source_failures'][0]['source_record']
        == (payload['benchmarks']['data'][0])
    )
    try:
        adapter.make_logs(fetched, base_url='https://example.test')
    except SourceRecordsError as exc:
        assert exc.source_name == 'LLM Stats'
        assert len(exc.failures) == 2
        assert all(
            failure.source_ref.startswith('https://example.test/')
            for failure in exc.failures
        )
    else:
        raise AssertionError('expected incomplete benchmark fetch to fail')


def aa_omniscience_payload() -> dict:
    detail = {
        'benchmark_id': 'aa-omniscience-index',
        'name': 'AA Omniscience Index',
        'description': 'Signed Artificial Analysis omniscience index.',
        'max_score': 200,
        'models': [
            {
                'model_id': 'grok-4-heavy',
                'model_name': 'Grok 4 Heavy',
                'organization_id': 'xai',
                'organization_name': 'xAI',
                'score': 126,
                'normalized_score': 0.63,
                'self_reported': False,
                'self_reported_source': 'https://artificialanalysis.ai/',
                'analysis_method': 'Raw signed index 26, shifted by 100.',
            },
            {
                'model_id': 'lfm2-24b-a2b',
                'model_name': 'LFM2-24B-A2B',
                'organization_id': 'liquid-ai',
                'organization_name': 'Liquid AI',
                'score': -29.5,
                'normalized_score': None,
                'self_reported': True,
                'self_reported_source': 'https://www.liquid.ai/',
                'analysis_method': 'Reported directly on signed scale.',
            },
        ],
    }
    return {
        'models': [],
        'benchmarks': [],
        'scores': adapter.scores_from_benchmark_detail(detail),
        'source_record_count': 2,
    }


def test_aa_omniscience_normalizes_mixed_live_scores_and_validates(
    tmp_path: Path,
):
    result = adapter.convert_logs(
        aa_omniscience_payload(), retrieved_timestamp='1234567890.0'
    )

    assert result.failures == []
    by_name = {
        bundle.log.model_info.name: bundle.log.evaluation_results[0]
        for bundle in result.records
    }
    grok = by_name['Grok 4 Heavy']
    liquid = by_name['LFM2-24B-A2B']

    assert grok.score_details.score == pytest.approx(26.0)
    assert liquid.score_details.score == pytest.approx(-29.5)
    for evaluation_result in (grok, liquid):
        assert evaluation_result.metric_config.min_score == -100
        assert evaluation_result.metric_config.max_score == 100
        assert evaluation_result.metric_config.metric_unit == 'points'

    assert grok.score_details.details['raw_score'] == '126'
    assert grok.score_details.details['raw_normalized_score'] == '0.63'
    assert grok.score_details.details['transformation_strategy'] == (
        'normalized_score_to_signed_range'
    )
    assert liquid.score_details.details['raw_score'] == '-29.5'
    assert liquid.score_details.details['raw_normalized_score'] == 'null'
    assert liquid.score_details.details['transformation_strategy'] == (
        'self_reported_signed_raw_score'
    )

    output_dir = tmp_path / 'data' / 'llm-stats'
    paths = adapter.export_logs(result.records, output_dir)
    assert len(paths) == 2
    for path in paths:
        report = validate_file(
            path,
            repo_path=str(path.relative_to(tmp_path)),
            available_files=frozenset(),
            run_semantic_checks=True,
        )
        assert report.valid, report.errors


def test_aa_omniscience_ambiguous_scale_is_a_row_failure():
    payload = aa_omniscience_payload()
    ambiguous = payload['scores'][0]
    ambiguous['normalized_score'] = None
    ambiguous['self_reported'] = False
    payload['scores'] = [ambiguous]
    payload['source_record_count'] = 1

    result = adapter.convert_logs(payload, retrieved_timestamp='1234567890.0')

    assert result.records == []
    assert len(result.failures) == 1
    assert 'cannot determine its score scale' in result.failures[0].reason
    assert result.failures[0].source_record == ambiguous


def test_generic_bounds_use_documented_max_and_reject_unbounded_negative():
    assert adapter.metric_bounds_and_unit(
        6.0, None, {'id': 'example', 'max_score': 10}
    ) == (0.0, 10.0, 'points', 'benchmark_max_with_zero_min')
    assert adapter.metric_bounds_and_unit(
        63.0, None, {'id': 'percent-example', 'max_score': 1}
    ) == (0.0, 100.0, 'percent', 'inferred_percent_from_score')
    with pytest.raises(ValueError, match='no documented adapter-specific'):
        adapter.metric_bounds_and_unit(-1.0, None, {'id': 'example'})


def vending_bench_2_payload() -> dict:
    benchmark = {
        'benchmark_id': 'vending-bench-2',
        'name': 'Vending-Bench 2',
        'description': 'Final bank balance after one simulated year.',
        'max_score': 1.0,
    }
    rows = [
        ('claude-opus-4-6', 'Claude Opus 4.6', 'anthropic', 8017.59),
        ('glm-5.1', 'GLM-5.1', 'zhipu-ai', 5634.41),
        ('gemini-3-pro-preview', 'Gemini 3 Pro', 'google', 5478.16),
        ('gemini-3-flash-preview', 'Gemini 3 Flash', 'google', 3635.0),
    ]
    return {
        'models': [],
        'benchmarks': [],
        'scores': [
            {
                'id': f'vending-bench-2::{model_id}',
                'model_id': model_id,
                'model_name': model_name,
                'organization_id': organization,
                'organization_name': organization,
                'score': score,
                'normalized_score': None,
                'self_reported': True,
                'source_url': adapter.VENDING_BENCH_2_SCALE_URL,
                'benchmark': benchmark,
            }
            for model_id, model_name, organization, score in rows
        ],
        'source_record_count': len(rows),
    }


def test_vending_bench_2_uses_unbounded_dollar_scale_and_validates(
    tmp_path: Path,
):
    result = adapter.convert_logs(
        vending_bench_2_payload(), retrieved_timestamp='1234567890.0'
    )

    assert result.failures == []
    scores = sorted(
        evaluation_result.score_details.score
        for bundle in result.records
        for evaluation_result in bundle.log.evaluation_results
    )
    assert scores == [3635.0, 5478.16, 5634.41, 8017.59]
    for bundle in result.records:
        evaluation_result = bundle.log.evaluation_results[0]
        metric = evaluation_result.metric_config
        assert metric.min_score == 0
        assert metric.max_score == float('inf')
        assert metric.metric_unit == 'usd'
        assert metric.additional_details['bound_strategy'] == (
            'vending_bench_2_unbounded_dollars'
        )
        assert metric.additional_details['raw_max_score'] == '1.0'
        assert metric.additional_details['canonical_scale_source_url'] == (
            adapter.VENDING_BENCH_2_SCALE_URL
        )

    output_dir = tmp_path / 'data' / 'llm-stats'
    paths = adapter.export_logs(result.records, output_dir)
    assert len(paths) == 4
    for path in paths:
        report = validate_file(
            path,
            repo_path=str(path.relative_to(tmp_path)),
            available_files=frozenset(),
            run_semantic_checks=True,
        )
        assert report.valid, report.errors


def test_community_benchmarks_are_excluded_without_fetching_details(
    monkeypatch, tmp_path: Path
):
    benchmarks = {
        'data': [
            {
                'id': 'gpqa',
                'name': 'GPQA',
                'max_score': 1,
                'is_community': False,
            },
            {
                'id': 'community:2256e9c9-b256-4444-b639-7cc3b1855d96',
                'name': 'nolima',
                'dataset_id': '2256e9c9-b256-4444-b639-7cc3b1855d96',
                'model_count': 52,
                'is_community': True,
            },
        ]
    }
    models = {
        'data': [
            {
                'id': 'gpt-5',
                'name': 'GPT-5',
                'provider': {'slug': 'openai', 'name': 'OpenAI'},
            }
        ]
    }
    fetched_urls: list[str] = []

    def fake_fetch(url, *, headers):
        del headers
        fetched_urls.append(url)
        if url.endswith('/v1/models'):
            return models
        if url.endswith('/leaderboard/benchmarks'):
            return benchmarks
        if url.endswith('/v1/scores'):
            raise adapter.FetchError('scores endpoint unavailable')
        if url.endswith('/leaderboard/benchmarks/gpqa'):
            return {
                'benchmark_id': 'gpqa',
                'name': 'GPQA',
                'max_score': 1,
                'models': [
                    {
                        'model_id': 'gpt-5',
                        'model_name': 'GPT-5',
                        'score': 0.9,
                        'self_reported': True,
                    }
                ],
            }
        raise AssertionError(f'unexpected fetch: {url}')

    monkeypatch.setattr(adapter, 'fetch_json', fake_fetch)
    monkeypatch.setattr(adapter, 'fetch_text', lambda _url: '')

    raw_path = tmp_path / 'raw' / 'combined.json'
    report_path = tmp_path / 'reports' / 'llm-stats.json'
    args = adapter.parse_args(
        [
            '--api-key',
            'secret',
            '--base-url',
            'https://example.test',
            '--output-dir',
            str(tmp_path / 'data' / 'llm-stats'),
            '--save-raw-json',
            str(raw_path),
            '--failure-report',
            str(report_path),
        ]
    )

    assert adapter.run(args) == 1
    assert not any(
        'community%3A' in url or 'community:' in url for url in fetched_urls
    )

    captured = json.loads(raw_path.read_text(encoding='utf-8'))
    assert captured['source_record_count'] == 2
    assert len(captured['source_exclusions']) == 1
    assert (
        captured['source_exclusions'][0]['source_record']['model_count'] == 52
    )

    report = json.loads(report_path.read_text(encoding='utf-8'))
    assert report['total_source_records'] == 2
    assert report['failed_record_count'] == 0
    assert report['excluded_record_count'] == 1


def test_export_paths_follow_datastore_layout(tmp_path: Path):
    output_dir = tmp_path / 'data' / 'llm-stats'
    bundles = adapter.make_logs(
        sample_payload(), retrieved_timestamp='1234567890.0'
    )
    paths = adapter.export_logs(bundles, output_dir)

    assert len(paths) == 3
    for path in paths:
        assert path.suffix == '.json'
        assert path.parent.parent.parent == output_dir
        report = validate_file(path)
        assert report.valid, report.errors

    assert (output_dir / 'openai' / 'gpt-5').is_dir()
    assert (output_dir / 'anthropic' / 'claude-4-opus').is_dir()


def test_scores_from_live_benchmark_detail_shape():
    detail = {
        'benchmark_id': 'gpqa',
        'name': 'GPQA',
        'description': 'Graduate-level science questions.',
        'max_score': 1.0,
        'models': [
            {
                'rank': 1,
                'model_id': 'gpt-5.5',
                'model_name': 'GPT-5.5',
                'organization_id': 'openai',
                'organization_name': 'OpenAI',
                'score': 0.936,
                'normalized_score': 0.936,
                'verified': False,
                'self_reported': True,
                'self_reported_source': 'https://openai.com/index/introducing-gpt-5-5/',
                'analysis_method': 'GPQA Diamond. Reasoning effort xhigh.',
            }
        ],
    }

    scores = adapter.scores_from_benchmark_detail(detail)

    assert len(scores) == 1
    assert scores[0]['benchmark']['benchmark_id'] == 'gpqa'
    assert scores[0]['model_id'] == 'gpt-5.5'
    assert (
        scores[0]['source_url']
        == 'https://openai.com/index/introducing-gpt-5-5/'
    )
    assert adapter.relationship_from_score(scores[0]) == 'first_party'


def test_extracts_model_page_score_sources():
    page_html = (
        r'{\"benchmark_id\":\"arc-agi-v2\",\"name\":\"ARC-AGI v2\",'
        r'\"score\":0.065,\"self_reported\":false,'
        r'\"self_reported_source\":\"https://x.com/xai/status/1943158495588815072\"}'
        r'{\"benchmark_id\":\"gpqa\",\"name\":\"GPQA\",'
        r'\"score\":0.936,\"self_reported\":true,'
        r'\"self_reported_source\":\"https://openai.com/index/introducing-gpt-5-5/\"}'
    )

    sources = adapter.extract_model_page_score_sources(page_html)

    assert sources['arc-agi-v2']['self_reported'] is False
    assert (
        sources['arc-agi-v2']['self_reported_source']
        == 'https://x.com/xai/status/1943158495588815072'
    )
    assert sources['arc-agi-v2']['source_organization'] == 'xai'
    assert sources['gpqa']['self_reported'] is True
    assert sources['gpqa']['source_organization'] == 'openai'


def test_enrich_scores_with_model_page_sources(monkeypatch):
    page_html = (
        r'{\"benchmark_id\":\"arc-agi-v2\",\"name\":\"ARC-AGI v2\",'
        r'\"score\":0.065,\"self_reported\":false,'
        r'\"self_reported_source\":\"https://x.com/xai/status/1943158495588815072\"}'
    )
    monkeypatch.setattr(adapter, 'fetch_text', lambda _url: page_html)
    scores = [
        {
            'model_id': 'o3-2025-04-16',
            'benchmark_id': 'arc-agi-v2',
            'score': 0.065,
        }
    ]

    enriched = adapter.enrich_scores_with_model_page_sources(scores)

    assert enriched[0]['self_reported'] is False
    assert (
        enriched[0]['source_url']
        == 'https://x.com/xai/status/1943158495588815072'
    )
    assert enriched[0]['source_organization'] == 'xai'


def test_model_page_failure_keeps_scores_and_records_provenance(monkeypatch):
    scores = [
        {
            'model_id': 'gpt-5.5',
            'benchmark_id': 'gpqa',
            'score': 0.936,
        }
    ]

    def fail_fetch(_url):
        raise adapter.FetchError('model page unavailable')

    monkeypatch.setattr(adapter, 'fetch_text', fail_fetch)

    result = adapter.enrich_scores_with_model_page_sources_result(scores)

    assert result.records == scores
    assert len(result.failures) == 1
    assert result.failures[0].source_ref.endswith('/models/gpt-5.5')
    assert result.failures[0].source_record == scores


def test_relationship_uses_score_source_against_model_developer():
    openai_model = {
        'id': 'o3-2025-04-16',
        'name': 'o3',
        'organization_id': 'openai',
        'organization_name': 'OpenAI',
    }

    assert (
        adapter.relationship_from_score(
            {'source_url': 'https://openai.com/index/o3/', 'score': 0.8},
            openai_model,
        )
        == 'first_party'
    )
    assert (
        adapter.relationship_from_score(
            {
                'source_url': 'https://x.com/xai/status/1943158495588815072',
                'score': 0.065,
            },
            openai_model,
        )
        == 'third_party'
    )
    assert (
        adapter.relationship_from_score(
            {'self_reported': False, 'score': 0.065},
            openai_model,
        )
        == 'third_party'
    )
    assert (
        adapter.relationship_from_score({'score': 0.065}, openai_model)
        == 'other'
    )


def test_scores_from_live_benchmark_detail_handles_empty_model_id():
    detail = {
        'benchmark_id': 'gpqa',
        'name': 'GPQA',
        'models': [
            {
                'model_id': None,
                'score': 0.936,
            }
        ],
    }

    scores = adapter.scores_from_benchmark_detail(detail)

    assert scores[0]['id'] == 'gpqa::unknown'


def test_benchmark_detail_result_keeps_valid_score_and_reports_bad_entry():
    valid_entry = {
        'model_id': 'gpt-5.5',
        'model_name': 'GPT-5.5',
        'score': 0.936,
    }
    bad_entry = {
        'model_id': 'broken-model',
        'model_name': 'Broken Model',
    }
    detail = {
        'benchmark_id': 'gpqa',
        'name': 'GPQA',
        'models': [valid_entry, bad_entry],
    }

    result = adapter.scores_from_benchmark_detail_result(detail)

    assert len(result.records) == 1
    assert result.records[0]['model_id'] == 'gpt-5.5'
    assert len(result.failures) == 1
    assert result.failures[0].source_ref == "benchmark 'gpqa' score row 1"
    assert result.failures[0].source_record == bad_entry


def test_live_benchmark_scores_preserve_score_level_organization():
    detail = {
        'benchmark_id': 'gpqa',
        'name': 'GPQA',
        'models': [
            {
                'model_id': 'gpt-5.5',
                'model_name': 'GPT-5.5',
                'organization_id': 'openai',
                'organization_name': 'OpenAI',
                'score': 0.936,
                'self_reported': True,
                'self_reported_source': 'https://openai.com/index/introducing-gpt-5-5/',
            }
        ],
    }
    payload = {
        'models': [],
        'benchmarks': [],
        'scores': adapter.scores_from_benchmark_detail(detail),
    }

    bundles = adapter.make_logs(payload, retrieved_timestamp='1234567890.0')

    assert len(bundles) == 1
    assert bundles[0].developer == 'openai'
    assert bundles[0].model == 'gpt-5.5'
    assert bundles[0].log.model_info.id == 'openai/gpt-5.5'


def test_relationship_accepts_canonical_values_from_provenance_keys():
    assert (
        adapter.relationship_from_score({'relationship': 'collaborative'})
        == 'collaborative'
    )
    assert (
        adapter.relationship_from_score({'verification_tier': 'third_party'})
        == 'third_party'
    )


def test_missing_model_and_benchmark_identity_fails_with_count():
    payload = {
        'models': [],
        'benchmarks': [],
        'scores': [{'score': 0.5}],
    }

    try:
        adapter.make_logs(payload, retrieved_timestamp='1234567890.0')
    except ValueError as exc:
        assert (
            'encountered 1 conversion issue(s) across 1 source record(s)'
            in str(exc)
        )
        assert 'model identity is required' in str(exc)
    else:
        raise AssertionError('expected missing identities to fail')
