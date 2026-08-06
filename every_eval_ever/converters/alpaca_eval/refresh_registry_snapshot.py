"""Refresh the vendored eval-card-registry snapshot this adapter resolves against.

Run by a maintainer, not by tests or CI::

    uv run python -m every_eval_ever.converters.alpaca_eval.refresh_registry_snapshot

The registry (``https://evaleval-entity-registry.hf.space``) is the shared
canonicalization service for EEE. This script reads its **read-only list
endpoints** and writes the snapshot that
:mod:`every_eval_ever.converters.alpaca_eval.registry` loads offline, so a
conversion is deterministic, needs no network, and cannot write to a shared
registry as a side effect of reading it.

That last point is not hypothetical: ``POST /api/v1/resolve`` defaults to
``mode="resolve"``, which **auto-creates a draft canonical** for anything it
cannot place, so bulk-resolving a leaderboard would silently add hundreds of
draft models. Only ``mode="exact"`` is side-effect-free, and the adapter's
opt-in live path uses it (see ``registry.py``). This script sticks to GETs.

``--check`` verifies the committed snapshot still matches the registry without
writing anything, so drift can be caught in review.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests

from every_eval_ever.converters.alpaca_eval.registry import (
    SNAPSHOT_PATH,
    normalize,
    snapshot_gaps,
)

REGISTRY_BASE_URL = 'https://evaleval-entity-registry.hf.space'
REQUEST_TIMEOUT = 180

#: Metric column names this adapter publishes, in the spelling the leaderboard
#: CSV uses. Only ``win_rate`` has a canonical entry today; the rest are
#: recorded as gaps so the adapter can say so rather than guess.
METRIC_QUERIES = (
    'win_rate',
    'length_controlled_winrate',
    'discrete_win_rate',
    'avg_length',
)

#: Benchmark names to look for, per leaderboard version.
BENCHMARK_QUERIES = ('AlpacaEval 1.0', 'AlpacaEval 2.0')

#: Harness name of the upstream library.
HARNESS_QUERIES = ('alpaca_eval',)


def _get(endpoint: str, base_url: str, **params: Any) -> List[Dict[str, Any]]:
    """Read one registry list endpoint and return its records."""
    response = requests.get(
        f'{base_url}/api/v1/{endpoint}', params=params, timeout=REQUEST_TIMEOUT
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise ValueError(
            f'expected a JSON array from {endpoint}, '
            f'got {type(payload).__name__}'
        )
    return payload


def _pick(
    records: Iterable[Dict[str, Any]], queries: Iterable[str]
) -> Dict[str, Optional[Dict[str, Any]]]:
    """Match each query against canonical ids, display names and aliases.

    Matching is punctuation- and case-insensitive (:func:`normalize`), which is
    how ``win_rate`` reaches the canonical ``win-rate``. A query that matches
    nothing maps to ``None`` and is kept in the snapshot: an explicit "the
    registry has no canonical for this" is what lets the adapter mark the metric
    unverified instead of silently attaching a wrong id.
    """
    index: Dict[str, Dict[str, Any]] = {}
    for record in records:
        keys = [record.get('id'), record.get('display_name')]
        keys.extend(record.get('aliases') or [])
        for key in keys:
            if isinstance(key, str) and normalize(key):
                index.setdefault(normalize(key), record)
    return {query: index.get(normalize(query)) for query in queries}


def _metric_entry(record: Dict[str, Any]) -> Dict[str, Any]:
    """Keep the fields a score depends on, plus how vetted the entry is.

    ``min_score``/``max_score`` are the reason to consult the registry at all —
    they settle the scale a win rate is published on — and ``review_status``
    travels with them so the adapter can surface (never silently trust) a
    still-``draft`` bound.
    """
    return {
        'id': record['id'],
        'display_name': record.get('display_name'),
        'score_type': record.get('score_type'),
        'lower_is_better': record.get('lower_is_better'),
        'min_score': record.get('min_score'),
        'max_score': record.get('max_score'),
        'review_status': record.get('review_status'),
    }


def _named_entry(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'id': record['id'],
        'display_name': record.get('display_name'),
        'review_status': record.get('review_status'),
    }


def build_snapshot(base_url: str = REGISTRY_BASE_URL) -> Dict[str, Any]:
    """Derive the offline vocabulary from the registry's list endpoints."""
    orgs = _get('orgs', base_url)
    org_aliases = _get('aliases', base_url, entity_type='org')
    metrics = _get('metrics', base_url)
    benchmarks = _get('benchmarks', base_url)
    harnesses = _get('harnesses', base_url)

    # ``hf_org`` is how a canonical org id reaches the HuggingFace namespace
    # models are actually published under (``meta`` -> ``meta-llama``,
    # ``alibaba`` -> ``qwen``, ``zai`` -> ``zai-org``). Both spellings are real
    # identities for the same organization, so the adapter needs the mapping in
    # both directions: the namespace is the model id prefix, the canonical id is
    # ``model_info.developer``.
    identities: Dict[str, str] = {}
    review_status: Dict[str, str] = {}
    for record in orgs:
        org_id = record.get('id')
        if not isinstance(org_id, str) or not normalize(org_id):
            continue
        identities[normalize(org_id)] = org_id
        if record.get('review_status'):
            review_status[org_id] = record['review_status']
    for record in orgs:
        namespace, org_id = record.get('hf_org'), record.get('id')
        if isinstance(namespace, str) and isinstance(org_id, str):
            identities.setdefault(normalize(namespace), org_id)

    # Only confirmed aliases, and only where they add a spelling the identities
    # above do not already cover — an unconfirmed alias is a guess, and this
    # snapshot is used to *decide* a published developer id.
    aliases: Dict[str, str] = {}
    for record in org_aliases:
        raw, canonical = record.get('raw_value'), record.get('canonical_id')
        if record.get('status') != 'confirmed':
            continue
        if not isinstance(raw, str) or not isinstance(canonical, str):
            continue
        key = normalize(raw)
        if key and key not in identities:
            aliases.setdefault(key, canonical)

    return {
        '_meta': {
            'source': f'{base_url}/api/v1 read-only list endpoints',
            'endpoints': [
                'orgs',
                'aliases?entity_type=org',
                'metrics',
                'benchmarks',
                'harnesses',
            ],
            'note': (
                'Vendored snapshot of eval-card-registry canonical entries. '
                'Regenerate with refresh_registry_snapshot.py. Do not edit by '
                'hand. Authoritative at snapshot time: entries added to the '
                'registry later resolve here only after a refresh.'
            ),
            'retrieved_date': datetime.now(timezone.utc)
            .date()
            .isoformat(),
            'counts': {
                'orgs': len(orgs),
                'org_aliases_confirmed': len(aliases),
                'org_identities': len(identities),
                'metrics': len(metrics),
                'benchmarks': len(benchmarks),
                'harnesses': len(harnesses),
            },
        },
        'org_identities': dict(sorted(identities.items())),
        'org_aliases': dict(sorted(aliases.items())),
        'org_review_status': dict(sorted(review_status.items())),
        'metrics': {
            query: (_metric_entry(record) if record else None)
            for query, record in _pick(metrics, METRIC_QUERIES).items()
        },
        'benchmarks': {
            query: (_named_entry(record) if record else None)
            for query, record in _pick(benchmarks, BENCHMARK_QUERIES).items()
        },
        'harnesses': {
            query: (_named_entry(record) if record else None)
            for query, record in _pick(harnesses, HARNESS_QUERIES).items()
        },
    }


def _serialize(snapshot: Dict[str, Any]) -> str:
    return json.dumps(snapshot, indent=2, sort_keys=False) + '\n'


def _comparable(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """The snapshot minus fields that change on every run."""
    trimmed = dict(snapshot)
    meta = dict(trimmed.get('_meta') or {})
    meta.pop('retrieved_date', None)
    trimmed['_meta'] = meta
    return trimmed


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default=REGISTRY_BASE_URL)
    parser.add_argument('--output', type=Path, default=SNAPSHOT_PATH)
    parser.add_argument(
        '--check',
        action='store_true',
        help='compare against the committed snapshot without writing',
    )
    args = parser.parse_args(argv)

    snapshot = build_snapshot(args.base_url)
    if args.check:
        if not args.output.exists():
            print(f'{args.output} does not exist', file=sys.stderr)
            return 1
        committed = json.loads(args.output.read_text(encoding='utf-8'))
        if _comparable(committed) == _comparable(snapshot):
            print(f'{args.output.name} is up to date')
            return 0
        print(
            f'{args.output.name} differs from the registry; rerun without '
            '--check to refresh',
            file=sys.stderr,
        )
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(_serialize(snapshot), encoding='utf-8')
    counts = snapshot['_meta']['counts']
    print(f'wrote {args.output} ({counts})')
    gaps = snapshot_gaps(snapshot)
    if gaps:
        print('no canonical entry for: ' + ', '.join(gaps))
    return 0


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())
