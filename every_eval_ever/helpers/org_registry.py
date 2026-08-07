"""Offline organization vocabulary from the eval-card-registry.

The registry (``https://evaleval-entity-registry.hf.space``) is the shared
canonicalization service for EEE. This module reads a snapshot of its two
read-only list endpoints, bundled as package data, rather than calling
``POST /api/v1/resolve``, which *creates* a ``draft`` canonical for anything it
cannot place — a validator must not write to a shared registry to check a file.
Refresh with ``python -m every_eval_ever.tools.refresh_org_registry``.

A slug is one of two things here:

- an **identity** — a canonical organization id, or a HuggingFace namespace the
  registry records for one. ``meta`` and ``meta-llama`` are both Meta, and a
  model is published under the namespace, so it is not drift. Only recorded
  namespaces count, so widening this means filling in ``hf_org`` upstream.
- a **second name** — a confirmed alias that is a genuinely different name for
  an organization already in the registry: ``AI2`` for ``allenai``, or a model
  family such as ``glm`` standing in for its publisher.

Anything else is unknown to the registry and gets no opinion from this module.
The canonical list is taken as-is, ``draft`` entries included; this vocabulary
only ever holds a warning back, so the untouched list is the conservative one.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from importlib import resources
from typing import Any, NamedTuple

#: Package-data file under ``helpers/data/``, read through
#: :func:`load_org_snapshot`. Only the refresh tool needs a filesystem path.
SNAPSHOT_NAME = 'org_registry.json'


class OrgVocabulary(NamedTuple):
    """Normalized lookups over one registry snapshot.

    Both are keyed by :func:`normalize_org_slug` output, so a lookup is
    insensitive to case and to ``-``/``_``/``.``/space punctuation. That is
    lossy — ``DeepAuto-AI`` and ``deepautoai`` are two canonical ids collapsing
    to one key — so ``identities`` is a set, answering only "is this spelling
    already a name of record". ``load_org_snapshot`` has the unnormalized ids.
    """

    identities: frozenset[str]
    second_names: dict[str, str]


def normalize_org_slug(value: str) -> str:
    """Collapse an organization slug to its punctuation-insensitive identity.

    ``moonshot-ai``, ``Moonshot AI`` and ``moonshotai`` all normalize alike: the
    registry aims for HuggingFace-true casing and HuggingFace is not consistent,
    so these are one name rather than three.
    """
    return re.sub(r'[^a-z0-9]+', '', value.strip().lower())


def load_org_snapshot() -> dict[str, Any]:
    """Return the bundled registry snapshot as parsed JSON."""
    resource = resources.files('every_eval_ever.helpers').joinpath(
        'data', SNAPSHOT_NAME
    )
    return json.loads(resource.read_text(encoding='utf-8'))


@lru_cache(maxsize=1)
def org_vocabulary() -> OrgVocabulary:
    """Return the normalized lookups built from the bundled snapshot."""
    snapshot = load_org_snapshot()
    identities = frozenset(
        normalize_org_slug(value)
        for value in (*snapshot['orgs'], *snapshot['hf_orgs'])
    )
    second_names = {
        normalize_org_slug(raw_value): org_id
        for raw_value, org_id in snapshot['second_names'].items()
        if normalize_org_slug(raw_value) not in identities
    }
    return OrgVocabulary(identities=identities, second_names=second_names)


def second_name_of(slug: str) -> str | None:
    """Return the canonical id when this slug is a *second name* for it.

    ``None`` for a canonical id, for a HuggingFace namespace the registry
    records, and for anything the registry has never seen.
    """
    if not isinstance(slug, str):
        return None
    vocabulary = org_vocabulary()
    normalized = normalize_org_slug(slug)
    if not normalized or normalized in vocabulary.identities:
        return None
    return vocabulary.second_names.get(normalized)
