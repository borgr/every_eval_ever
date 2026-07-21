"""Offline, fixture-based tests for the Papers with Code adapter.

No network and no PostgreSQL: the pure builders take plain dicts (the same shape
``pgdumplib`` yields), so the ``core`` CI matrix runs these without ``pgdumplib``.
"""

from __future__ import annotations

import json

from every_eval_ever.validate import validate_file
from utils.paperswithcode import adapter

RETRIEVED_TS = '1700000000.0'
DUMP_VERSION = '20260716'


def _datasets():
    return {
        # hf_dataset source
        '218': {
            'id': '218',
            'name': 'ETH3D (relative)',
            'slug': 'eth3d-relative',
            'hf_url': 'https://huggingface.co/datasets/ritianyu/eth3d_tar',
            'url': 'https://www.eth3d.net/schoeps2017cvpr.pdf',
            'paper_url': 'https://arxiv.org/abs/1704.00001',
            'introduced_year': '2017',
        },
        # url source (no hf_url)
        '906': {
            'id': '906',
            'name': 'RealEstate10K (2-view)',
            'slug': 're10k-2-view',
            'hf_url': None,
            'url': 'https://google.github.io/realestate10k/',
            'paper_url': None,
        },
        # private/other source (no urls and no slug -> unreachable in real PwC
        # data, but the code path must still produce a valid record)
        '999': {
            'id': '999',
            'name': 'Secret Bench',
            'slug': None,
            'hf_url': None,
            'url': None,
            'paper_url': None,
        },
    }


def _tasks():
    return {
        '10': {'id': '10', 'slug': 'depth-estimation'},
        '20': {'id': '20', 'slug': 'novel-view-synthesis'},
        '30': {'id': '30', 'slug': 'secret-task'},
    }


def _metric_dir():
    return {
        'AbsRel': 'lower_is_better',
        'delta1': 'higher_is_better',
        'PSNR': 'higher_is_better',
        'SSIM': 'higher_is_better',
        'Accuracy': 'higher_is_better',
        'CustomZMetric': 'higher_is_better',  # not in the registry snapshot
    }


def _metric_meta():
    return {
        'AbsRel': {'full_name': 'Absolute Relative Error', 'scale': '0-1'},
        'delta1': {'full_name': 'Delta < 1.25', 'scale': '0-1'},
        'PSNR': {'full_name': 'Peak Signal-to-Noise Ratio', 'scale': 'unbounded'},
        'Accuracy': {'full_name': 'Accuracy', 'scale': None},
    }


def _metric_ranges():
    return {
        'AbsRel': (0.008, 0.794),
        'delta1': (0.629, 0.993),
        'PSNR': (13.5, 41.22),
        'SSIM': (0.39, 0.9831),
        'Accuracy': (0.0, 100.0),
        'CustomZMetric': (0.0, 1.0),
    }


def _evaluations():
    return [
        # open model (hf url), multi-metric row: delta1 + SSIM (both bounded) for
        # fan-out, plus AbsRel (unbounded -> emitted with inf)
        {
            'id': '11533',
            'paper_id': '900',
            'task_id': '10',
            'dataset_id': '218',
            'model_name': 'MoGe-2',
            'metrics': json.dumps(
                {'delta1': '0.991', 'SSIM': '0.85', 'AbsRel': '0.028'}
            ),
            'evaluated_on': '2026-07-06',
            'created_at': '2026-06-18 13:13:06+00',
            'hf_model_url': 'https://huggingface.co/Ruicheng/moge-2-vitl',
            'num_parameters': '326000000',
            'is_open': 't',
            'external': 'f',
            'harness': None,
            'best_rank': '3',
        },
        # closed/unknown-developer model, url dataset, PSNR (unbounded -> inf) +
        # a made-up metric that is not in the registry (unresolved path).
        {
            'id': '13524',
            'paper_id': None,
            'task_id': '20',
            'dataset_id': '906',
            'model_name': 'StructSplat',
            'metrics': json.dumps({'PSNR': '22.240', 'CustomZMetric': '0.5'}),
            'evaluated_on': None,
            'created_at': '2026-07-14 00:00:00+00',
            'hf_model_url': None,
            'is_open': 't',
            'external': 'f',
            'harness': 'Not reported',
        },
        # European decimal comma + a known LLM name -> developer from helper
        {
            'id': '20001',
            'paper_id': None,
            'task_id': '10',
            'dataset_id': '218',
            'model_name': 'GPT-5.5 Pro (xhigh)',
            'metrics': json.dumps({'Accuracy': '97,3'}),
            'evaluated_on': None,
            'created_at': '2026-07-01 00:00:00+00',
            'hf_model_url': None,
            'is_open': 'f',
            'external': 't',
            'harness': 'SWE-agent',
        },
    ]


def _papers():
    return {
        '900': {
            'arxiv_id': '2507.02546',
            'title': 'MoGe-2',
            'source_url': None,
        }
    }


def _make_resolver():
    return adapter.MetricResolver(pwc_directions=_metric_dir())


def _build(resolver=None):
    resolver = resolver or _make_resolver()
    return adapter.build_logs(
        _evaluations(),
        _datasets(),
        _tasks(),
        resolver,
        _metric_ranges(),
        _metric_meta(),
        _papers(),
        DUMP_VERSION,
        RETRIEVED_TS,
    )


def test_parse_metric_value_edge_cases():
    assert adapter.parse_metric_value('95.2') == (95.2, None)
    assert adapter.parse_metric_value('30%') == (30.0, None)
    # European decimal comma, not thousands separator
    assert adapter.parse_metric_value('97,3') == (97.3, None)
    # thousands separator
    assert adapter.parse_metric_value('1,234.5') == (1234.5, None)
    # uncertainty
    score, se = adapter.parse_metric_value('33.7 ± 0.82')
    assert score == 33.7 and se == 0.82
    assert adapter.parse_metric_value('n/a') == (None, None)


def test_model_identity_prefers_hf_url_casing():
    mid, dev, slug, name = adapter.model_identity(
        'MoGe-2', 'https://huggingface.co/Ruicheng/moge-2-vitl'
    )
    assert mid == 'Ruicheng/moge-2-vitl'  # HF-true casing preserved
    assert dev == 'Ruicheng'


def test_model_identity_guesses_developer_from_name():
    mid, dev, slug, name = adapter.model_identity('GPT-5.5 Pro (xhigh)', None)
    assert dev == 'openai'
    assert name == 'GPT-5.5 Pro (xhigh)'  # raw display name preserved


def test_source_data_variants():
    ds = _datasets()
    hf = adapter.build_source_data(ds['218'])
    assert hf.source_type == 'hf_dataset' and hf.hf_repo == 'ritianyu/eth3d_tar'
    url = adapter.build_source_data(ds['906'])
    assert url.source_type == 'url' and url.url
    other = adapter.build_source_data(ds['999'])
    assert other.source_type == 'other'


def test_evaluation_id_is_stable_not_now():
    bundles = _build()
    for b in bundles:
        assert b.log.evaluation_id.endswith(f'/{DUMP_VERSION}')
        assert RETRIEVED_TS not in b.log.evaluation_id


def test_directions_and_bounds():
    bundles = _build()
    by_metric = {}
    for b in bundles:
        for r in b.log.evaluation_results:
            by_metric[r.metric_config.metric_name] = r.metric_config
    # bounded canonical metrics are emitted with finite [0,1] bounds + direction
    assert by_metric['delta1'].lower_is_better is False
    assert by_metric['delta1'].max_score == 1.0
    assert by_metric['SSIM'].max_score == 1.0
    # unbounded metrics (PSNR, AbsRel) are emitted with inf (serialized "Infinity")
    assert by_metric['PSNR'].max_score == float('inf')
    assert by_metric['AbsRel'].max_score == float('inf')
    assert by_metric['AbsRel'].lower_is_better is True


def test_multi_metric_row_fans_out_with_distinct_result_ids():
    bundles = _build()
    moge = next(b for b in bundles if b.log.model_info.id == 'Ruicheng/moge-2-vitl')
    ids = [r.evaluation_result_id for r in moge.log.evaluation_results]
    # delta1 + SSIM + AbsRel all fan out (AbsRel emitted with an inf bound)
    assert set(ids) == {
        'paperswithcode.11533.delta1',
        'paperswithcode.11533.ssim',
        'paperswithcode.11533.absrel',
    }


def test_unbounded_metrics_emitted_with_inf_and_reported():
    resolver = _make_resolver()
    _build(resolver)
    # PSNR (row 13524) and AbsRel (row 11533) are unbounded in the registry
    assert set(resolver.unbounded_emitted) == {'PSNR', 'AbsRel'}


def test_additional_details_are_all_strings():
    bundles = _build()
    for b in bundles:
        for d in (b.log.model_info.additional_details or {}).values():
            assert isinstance(d, str)
        for r in b.log.evaluation_results:
            for d in (r.score_details.details or {}).values():
                assert isinstance(d, str)


def test_built_logs_validate(tmp_path):
    """Prove the skeleton: construct, save, and run the real validator."""
    bundles = _build()
    assert bundles
    for b in bundles:
        path = adapter.save_evaluation_log(
            b.log, tmp_path, b.developer, b.model
        )
        report = validate_file(path)
        assert report.valid, report.errors


# --- registry resolver / three tiers -------------------------------------------


def test_resolver_registry_hit_uses_canonical():
    r = _make_resolver()
    m = r.resolve('Accuracy', (0.0, 1.0))  # source already on canonical scale
    assert m.resolved is True
    assert m.metric_id == 'accuracy'
    assert (m.min_score, m.max_score) == (0.0, 1.0)  # from the registry, not 0.39
    assert m.lower_is_better is False
    assert not r.unresolved


def test_resolver_keeps_canonical_bounds():
    r = _make_resolver()
    # canonical accuracy is [0,1]; the resolver keeps that regardless of obs range
    m = r.resolve('Accuracy', (0.0, 100.0))
    assert m.resolved is True and m.metric_id == 'accuracy'
    assert (m.min_score, m.max_score) == (0.0, 1.0)


def test_reconcile_scale_percent_to_proportion():
    # PwC percent score rescaled onto canonical [0,1]
    score, se, detail = adapter.reconcile_scale(97.3, None, 0.0, 1.0, resolved=True)
    assert score == 0.973 and detail['canonical_rescale_factor'] == 100.0
    # std_err rescales with the score (same units)
    _, se2, _ = adapter.reconcile_scale(50.0, 2.0, 0.0, 1.0, resolved=True)
    assert se2 == 0.02
    # already on the canonical scale -> untouched
    assert adapter.reconcile_scale(0.87, None, 0.0, 1.0, resolved=True) == (
        0.87, None, {},
    )
    # unresolved metrics are never rescaled (bounds are observed, same scale)
    assert adapter.reconcile_scale(97.3, None, 0.0, 100.0, resolved=False) == (
        97.3, None, {},
    )


def test_resolver_unbounded_canonical_emits_inf():
    r = _make_resolver()
    # 'elo' is unbounded in the registry (max_score: null)
    m = r.resolve('ELO', (900.0, 2100.0))
    assert m.resolved is True and m.metric_id == 'elo'
    # null bound -> inf (serialized as "Infinity" per every_eval_ever#207)
    assert m.max_score == float('inf')
    assert m.detail['canonical_max'] == 'unbounded'


def test_accuracy_rescaled_to_canonical_in_build():
    accs = [
        r
        for b in _build()
        for r in b.log.evaluation_results
        if r.metric_config.metric_id == 'accuracy'
    ]
    assert accs
    for r in accs:
        assert r.metric_config.max_score == 1.0
        assert 0.0 <= r.score_details.score <= 1.0
        assert r.score_details.details.get('canonical_rescale_factor') == '100.0'


def test_resolver_unresolved_is_recorded_and_falls_back():
    r = _make_resolver()
    m = r.resolve('MadeUpMetricXYZ', (0.0, 5.0), dataset_slug='some-bench')
    assert m.resolved is False
    assert m.metric_id == 'paperswithcode.madeupmetricxyz'
    assert 'some-bench' in r.unresolved['MadeUpMetricXYZ']


def test_fail_closed_report_names_metrics_and_next_step():
    resolver = _make_resolver()
    _build(resolver)  # CustomZMetric is not in the registry snapshot
    assert resolver.unresolved  # would trigger the fail-closed gate in run()
    msg = adapter._report_unresolved(resolver.unresolved)
    assert 'CustomZMetric' in msg and 'registry-entity-aliases' in msg
    assert '--allow-unresolved' in msg
