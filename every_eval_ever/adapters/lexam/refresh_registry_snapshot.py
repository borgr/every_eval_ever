"""Regenerate `registry_snapshot.json` from a local eval-card-registry checkout.

The snapshot vendors only the entities this adapter emits: model ids (subjects
and judges), metric ids with their canonical bounds, the harness and the
benchmark aliases. `tests/test_lexam_adapter.py` fails when any of them drifts
from the registry.

Usage:
    uv run python -m every_eval_ever.adapters.lexam.refresh_registry_snapshot \
        --registry /path/to/eval-card-registry
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml

from every_eval_ever.adapters.lexam.adapter import (
    _MODEL_IDENTITIES,
    JUDGE_MODEL_IDS,
    MCQ_CONFIG,
    MCQ_METRIC,
    OPEN_QUESTION_CONFIG,
    OPEN_QUESTION_METRIC,
    REGISTRY_HARNESS,
    SNAPSHOT_PATH,
)

METRIC_FIELDS = (
    'min_score',
    'max_score',
    'lower_is_better',
    'score_type',
    'review_status',
)


def _normalize(value: str) -> str:
    """Mirror the registry's `normalized` matcher: drop case and separators."""
    return re.sub(r'[^a-z0-9]', '', str(value).lower())


def _load(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding='utf-8'))


def build_snapshot(registry: Path) -> dict[str, Any]:
    seed = registry / 'seed'
    metrics = {e['id']: e for e in _load(seed / 'metrics.yaml')}
    harnesses = {e['id'] for e in _load(seed / 'harnesses.yaml')}
    benchmarks = _load(seed / 'benchmarks.yaml')
    core = _load(seed / 'models' / 'core.yaml')

    # Curated floor plus the generator sources, both of which are in git, so a
    # plain checkout is enough — no local `seed --local` build required.
    model_ids = {e['id'] for e in core['entries']}
    for source in sorted((seed / 'models' / 'sources').glob('*.yaml')):
        loaded = _load(source)
        entries = (
            loaded.get('entries', loaded)
            if isinstance(loaded, dict)
            else loaded
        )
        if isinstance(entries, list):
            model_ids |= {
                e['id'] for e in entries if isinstance(e, dict) and 'id' in e
            }
    model_ids -= set(core.get('skip_source_ids') or [])

    benchmark_alias: dict[str, str] = {}
    for entry in benchmarks:
        for form in [entry['id'], *(entry.get('aliases') or [])]:
            benchmark_alias[_normalize(form)] = entry['id']

    used_models = {i.model_id for i in _MODEL_IDENTITIES.values()}
    used_models |= set(JUDGE_MODEL_IDS)
    used_metrics = {MCQ_METRIC.metric_id, OPEN_QUESTION_METRIC.metric_id}
    used_benchmarks = {
        f'lexam.{OPEN_QUESTION_CONFIG}',
        f'lexam.{MCQ_CONFIG}',
    }

    revision = subprocess.run(
        ['git', '-C', str(registry), 'rev-parse', 'HEAD'],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    return {
        '_meta': {
            'source': (
                'evaleval/eval-card-registry seed/ (metrics, harnesses, '
                'benchmarks, models/core, models/sources)'
            ),
            'registry_revision': revision,
            'note': (
                'Only the entities this adapter emits. Regenerate with '
                'refresh_registry_snapshot.py; do not edit by hand. '
                '`models_absent_from_seed` are ids the live registry resolves '
                'but no checked-in seed file declares (build-only draft '
                'canonicals) - see curation/UPSTREAM_DATA_ISSUES.md.'
            ),
        },
        'models': sorted(used_models & model_ids),
        'models_absent_from_seed': sorted(used_models - model_ids),
        'metrics': {
            name: {field: metrics[name][field] for field in METRIC_FIELDS}
            for name in sorted(used_metrics)
            if name in metrics
        },
        'metrics_unresolved': sorted(
            name for name in used_metrics if name not in metrics
        ),
        'harnesses': sorted({REGISTRY_HARNESS} & harnesses),
        'harnesses_unresolved': sorted({REGISTRY_HARNESS} - harnesses),
        'benchmarks': {
            name: benchmark_alias.get(_normalize(name))
            for name in sorted(used_benchmarks)
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--registry',
        type=Path,
        required=True,
        help='Path to a local eval-card-registry checkout.',
    )
    parser.add_argument('--output', type=Path, default=SNAPSHOT_PATH)
    parser.add_argument(
        '--check',
        action='store_true',
        help=(
            'Exit non-zero if the committed snapshot differs from the given '
            'registry, without writing. Use this after a registry PR merges '
            'to see whether the pin is stale.'
        ),
    )
    args = parser.parse_args(argv)

    snapshot = build_snapshot(args.registry)
    rendered = json.dumps(snapshot, indent=2, sort_keys=True) + '\n'

    if args.check:
        current = (
            args.output.read_text(encoding='utf-8')
            if args.output.exists()
            else ''
        )
        if current == rendered:
            print(f'{args.output} matches {args.registry}')
            return 0
        committed_revision = 'missing'
        if current:
            committed_revision = (
                json.loads(current)
                .get('_meta', {})
                .get('registry_revision', '?')
            )
        print(
            f'{args.output} is stale: pinned at {committed_revision}, '
            f'registry is at {snapshot["_meta"]["registry_revision"]}. '
            'Re-run without --check to refresh.'
        )
        return 1

    args.output.write_text(rendered, encoding='utf-8')
    print(f'wrote {args.output}')
    for key in (
        'models_absent_from_seed',
        'metrics_unresolved',
        'harnesses_unresolved',
    ):
        if snapshot[key]:
            print(f'  {key}: {snapshot[key]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
