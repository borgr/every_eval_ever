"""Refresh the bundled organization vocabulary from the eval-card-registry.

Run by a maintainer, not by tests or CI:

    uv run python -m every_eval_ever.tools.refresh_org_registry

The registry is the shared canonicalization service for EEE
(``https://evaleval-entity-registry.hf.space``). This script reads its two
**read-only** list endpoints and writes the derived snapshot that
``every_eval_ever.helpers.org_registry`` loads offline.

It deliberately does not touch ``POST /api/v1/resolve``: resolving an unknown
value auto-creates a ``draft`` canonical, so a bulk resolve would write to a
shared registry as a side effect of reading it.

Use ``--check`` to verify the committed snapshot still matches the registry
without writing anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import requests

from every_eval_ever import helpers
from every_eval_ever.helpers.org_registry import (
    SNAPSHOT_NAME,
    normalize_org_slug,
)

#: The snapshot inside this checkout. Only this tool writes it, so only this
#: tool needs a filesystem path for it.
SNAPSHOT_PATH = Path(helpers.__file__).resolve().parent / 'data' / SNAPSHOT_NAME

REGISTRY_BASE_URL = 'https://evaleval-entity-registry.hf.space'
ORGS_ENDPOINT = '/api/v1/orgs'
ALIASES_ENDPOINT = '/api/v1/aliases?entity_type=org'
REQUEST_TIMEOUT = 120


def _get(base_url: str, endpoint: str) -> list[dict[str, Any]]:
    """Read one registry list endpoint and return its records."""
    response = requests.get(f'{base_url}{endpoint}', timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise ValueError(
            f'expected a JSON array from {endpoint}, got {type(payload).__name__}'
        )
    return payload


def build_snapshot(
    orgs: list[dict[str, Any]], aliases: list[dict[str, Any]]
) -> dict[str, Any]:
    """Derive the offline vocabulary from raw registry records.

    Three groups come out of this, and the split is what makes the vocabulary
    usable offline:

    - ``orgs`` — every canonical organization id.
    - ``hf_orgs`` — HuggingFace namespaces that differ from the canonical id
      (``meta-llama`` for ``meta``, ``qwen`` for ``alibaba``). These are real
      identities, not drift: a model id is published under the namespace.
    - ``second_names`` — confirmed aliases that are a genuinely different name
      for an organization the registry already knows (``AI2`` for ``allenai``,
      ``Mistral`` for ``mistralai``, model families such as ``glm`` used where
      their publisher belongs).

    Two rules decide what an alias has to clear, and both drop toward silence:

    - An identity wins. A spelling that normalizes onto a canonical id or a
      HuggingFace namespace is dropped, whether it normalizes onto *its own*
      organization (``Mistral AI`` for ``mistralai`` — no information) or onto a
      different one. The second case is the interesting one: the registry has
      ``ai21-labs`` and ``ai21`` as separate canonical ids and confirms
      ``AI21 Labs`` as an alias of ``ai21``, so the alias would claim a
      spelling another organization already answers to. Six publishers are in
      that position today (``ai21``/``ai21-labs``, ``ibm``/``ibm-granite``,
      ``inception``/``inceptionlabs``, ``LGAI-EXAONE``/``lg-ai``,
      ``internlm``/``shanghai-ai-lab``, ``LiquidAI``/``liquid``). Until the
      registry settles which id is primary, saying nothing is the only answer
      that cannot be wrong.
    - One spelling per normalized form. Aliases that disagree about the
      organization are dropped as ambiguous; otherwise the lexicographically
      smallest spelling is kept so a refresh diffs cleanly.
    """
    canonical_ids = sorted({str(org['id']) for org in orgs if org.get('id')})
    identity_norms = {normalize_org_slug(org_id) for org_id in canonical_ids}

    hf_orgs: dict[str, str] = {}
    for org in sorted(orgs, key=lambda org: str(org.get('id', ''))):
        hf_org = org.get('hf_org')
        if not isinstance(hf_org, str) or not hf_org.strip():
            continue
        normalized = normalize_org_slug(hf_org)
        if not normalized or normalized in identity_norms:
            continue
        hf_orgs[hf_org.strip()] = str(org['id'])
        identity_norms.add(normalized)

    known_canonical = set(canonical_ids)
    by_norm: dict[str, tuple[str, str]] = {}
    ambiguous: set[str] = set()
    for alias in aliases:
        if alias.get('status') != 'confirmed':
            continue
        raw_value = alias.get('raw_value')
        canonical_id = alias.get('canonical_id')
        if (
            not isinstance(raw_value, str)
            or canonical_id not in known_canonical
        ):
            continue
        normalized = normalize_org_slug(raw_value)
        if not normalized or normalized in identity_norms:
            continue
        existing = by_norm.get(normalized)
        if existing is not None:
            if existing[1] != canonical_id:
                ambiguous.add(normalized)
            # Keep one spelling per normalized form; the first sorted spelling
            # is arbitrary but stable across refreshes.
            if existing[0] <= raw_value.strip():
                continue
        by_norm[normalized] = (raw_value.strip(), str(canonical_id))
    second_names = {
        raw_value: canonical_id
        for normalized, (raw_value, canonical_id) in by_norm.items()
        if normalized not in ambiguous
    }

    return {
        '_source': REGISTRY_BASE_URL,
        '_endpoints': [ORGS_ENDPOINT, ALIASES_ENDPOINT],
        '_refresh': 'python -m every_eval_ever.tools.refresh_org_registry',
        '_note': (
            'Derived from the registry list endpoints, not a verbatim mirror: '
            'spellings that differ from an identity only by case or '
            'punctuation are dropped because matching is '
            'punctuation-insensitive, and aliases whose normalized spelling '
            'points at two organizations are dropped as ambiguous.'
        ),
        'orgs': canonical_ids,
        'hf_orgs': dict(sorted(hf_orgs.items())),
        'second_names': dict(sorted(second_names.items())),
    }


def render_snapshot(snapshot: dict[str, Any]) -> str:
    """Serialize the snapshot so refreshes produce reviewable diffs."""
    return json.dumps(snapshot, indent=2, ensure_ascii=False) + '\n'


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog='refresh_org_registry',
        description=(
            'Refresh the bundled eval-card-registry organization vocabulary.'
        ),
    )
    parser.add_argument(
        '--base-url',
        default=REGISTRY_BASE_URL,
        help=f'Registry base URL (default: {REGISTRY_BASE_URL})',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help=f'Snapshot path to write (default: {SNAPSHOT_PATH})',
    )
    parser.add_argument(
        '--check',
        action='store_true',
        help='Report whether the snapshot is stale instead of writing it',
    )
    args = parser.parse_args(argv)

    output = args.output or SNAPSHOT_PATH
    try:
        orgs = _get(args.base_url, ORGS_ENDPOINT)
        aliases = _get(args.base_url, ALIASES_ENDPOINT)
    except (requests.RequestException, ValueError) as exc:
        print(f'could not read the registry: {exc}', file=sys.stderr)
        return 1

    snapshot = build_snapshot(orgs, aliases)
    rendered = render_snapshot(snapshot)
    summary = (
        f'{len(orgs)} orgs and {len(aliases)} org aliases read; '
        f'snapshot has {len(snapshot["orgs"])} canonical ids, '
        f'{len(snapshot["hf_orgs"])} distinct HuggingFace namespaces, '
        f'{len(snapshot["second_names"])} second names'
    )

    if args.check:
        current = output.read_text(encoding='utf-8') if output.is_file() else ''
        if current == rendered:
            print(f'{output} is up to date: {summary}')
            return 0
        print(f'{output} is stale: {summary}', file=sys.stderr)
        return 1

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding='utf-8')
    print(f'wrote {output}: {summary}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
