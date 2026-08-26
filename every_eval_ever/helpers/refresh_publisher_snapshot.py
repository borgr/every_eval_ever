#!/usr/bin/env python3
"""Regenerate the vendored publisher snapshot from the eval-card-registry.

The validator resolves a publisher spelling against a committed snapshot of the
registry's canonical organizations (``publisher_snapshot.json``) so it needs no
registry install, and no network, to run. Regenerate this whenever the
registry's ``seed/orgs.yaml`` changes:

    uv run python -m every_eval_ever.helpers.refresh_publisher_snapshot \
        --seed ../eval-card-registry/seed/orgs.yaml

Only the curated ``seed/orgs.yaml`` is read, not ``orgs.generated.yaml``: the
generated file is HuggingFace community namespaces with no aliases, so it says
nothing about one publisher having two names, and 900 extra rows would make the
snapshot's own diffs unreadable.

A snapshot is authoritative-at-snapshot-time. A publisher whose second name the
registry learns later is not recognized here until a refresh, which is the safe
direction: the check warns on what it can prove and stays silent otherwise.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import yaml

DEFAULT_SEED = Path('../eval-card-registry/seed/orgs.yaml')
SNAPSHOT = Path(__file__).with_name('publisher_snapshot.json')


def _registry_revision(seed_path: Path) -> str | None:
    """Best-effort git revision of the registry the seed was read from.

    Recorded in the snapshot ``_meta`` so a reader can pin which registry commit
    a warning came from. A ``-dirty`` suffix flags an uncommitted working tree.
    Returns None if the seed is not inside a git checkout.
    """
    repo = seed_path.resolve().parent
    try:
        sha = subprocess.run(
            ['git', '-C', str(repo), 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None
    if not sha:
        return None
    status = subprocess.run(
        ['git', '-C', str(repo), 'status', '--porcelain'],
        capture_output=True,
        text=True,
    )
    return sha + ('-dirty' if status.stdout.strip() else '')


def build_snapshot(seed_path: Path) -> dict:
    rows = yaml.safe_load(seed_path.read_text(encoding='utf-8')) or []
    publishers = [
        {
            'id': row['id'],
            'display_name': row.get('display_name'),
            # The namespace the organization publishes under. Kept separate from
            # the aliases because models really are published there, so it is
            # not evidence of a split directory.
            'hf_org': row.get('hf_org'),
            'aliases': row.get('aliases') or [],
        }
        for row in rows
        if row.get('id')
    ]
    meta = {
        'source': 'eval-card-registry seed/orgs.yaml',
        'note': (
            'Vendored snapshot of canonical organizations and their second '
            'names. Regenerate with refresh_publisher_snapshot.py. Do not '
            'edit by hand.'
        ),
        'count': len(publishers),
    }
    revision = _registry_revision(seed_path)
    if revision:
        meta['registry_revision'] = revision
    return {'_meta': meta, 'publishers': publishers}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=Path, default=DEFAULT_SEED)
    ap.add_argument('--out', type=Path, default=SNAPSHOT)
    args = ap.parse_args()
    snapshot = build_snapshot(args.seed)
    args.out.write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    revision = snapshot['_meta'].get(
        'registry_revision', '(unknown — not a git checkout)'
    )
    print(
        f'wrote {args.out} with {snapshot["_meta"]["count"]} publishers '
        f'@ registry {revision}'
    )


if __name__ == '__main__':
    main()
