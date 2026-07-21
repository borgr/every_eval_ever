#!/usr/bin/env python3
"""Regenerate the vendored canonical-metric snapshot from the eval-card-registry.

The adapter resolves metric bounds/direction against a committed snapshot of the
registry's canonical metrics (``registry_metrics.json``) so it needs no registry
install at runtime. Run this (with the registry repo checked out) whenever the
registry's ``seed/metrics.yaml`` changes:

    python -m utils.paperswithcode.refresh_metric_snapshot \
        --seed ../eval-card-registry/seed/metrics.yaml

A snapshot is authoritative-at-snapshot-time; new metrics added to the registry
only resolve here after a refresh (until then the adapter fails closed on them).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

DEFAULT_SEED = Path('../eval-card-registry/seed/metrics.yaml')
SNAPSHOT = Path(__file__).with_name('registry_metrics.json')


def build_snapshot(seed_path: Path) -> dict:
    rows = yaml.safe_load(seed_path.read_text(encoding='utf-8'))
    metrics = [
        {
            'id': r['id'],
            'display_name': r.get('display_name'),
            'aliases': r.get('aliases') or [],
            'score_type': r.get('score_type'),
            'lower_is_better': r.get('lower_is_better'),
            'min_score': r.get('min_score'),
            'max_score': r.get('max_score'),
        }
        for r in rows
    ]
    return {
        '_meta': {
            'source': 'eval-card-registry seed/metrics.yaml',
            'note': (
                'Vendored snapshot of canonical metric entries. Regenerate with '
                'refresh_metric_snapshot.py. Do not edit by hand.'
            ),
            'count': len(metrics),
        },
        'metrics': metrics,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=Path, default=DEFAULT_SEED)
    ap.add_argument('--out', type=Path, default=SNAPSHOT)
    args = ap.parse_args()
    snap = build_snapshot(args.seed)
    args.out.write_text(
        json.dumps(snap, indent=2, ensure_ascii=False) + '\n', encoding='utf-8'
    )
    print(f'wrote {args.out} with {snap["_meta"]["count"]} canonical metrics')


if __name__ == '__main__':
    main()
