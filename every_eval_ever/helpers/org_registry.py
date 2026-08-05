"""Offline organization vocabulary from the eval-card-registry.

The registry (``https://evaleval-entity-registry.hf.space``) is the shared
canonicalization service for EEE. Its ``POST /api/v1/resolve`` endpoint answers
"which organization is this?" live, but it also *creates* a ``draft`` canonical
for anything it cannot place — so a validator cannot call it: checking a file
would write to a shared registry as a side effect.

This module reads a snapshot of the registry's two read-only list endpoints
instead, bundled as package data. That keeps validation offline,
deterministic, and free of write side effects. Refresh it with
``python -m every_eval_ever.tools.refresh_org_registry``.

The vocabulary distinguishes two things a slug can be, which is the whole point
of using the registry rather than a hand-kept map:

- an **identity** — a canonical organization id, or a HuggingFace namespace the
  registry records for one. ``meta`` and ``meta-llama`` are both identities for
  Meta, ``alibaba`` and ``qwen`` both for Alibaba. Models are published under
  the namespace, so it is not drift. Only the namespaces the registry has
  recorded count: it fills in ``hf_org`` for 11 of its 1166 organizations so
  far, so ``MiniMaxAI`` is a second name for ``minimax`` here rather than an
  identity, and widening that means filling in ``hf_org`` upstream.
- a **second name** — a confirmed alias that is a genuinely different name for
  an organization already in the registry: ``AI2`` for ``allenai``, ``Mistral``
  for ``mistralai``, or a model family such as ``glm`` standing in for its
  publisher.

Anything else is unknown to the registry and gets no opinion from this module.

The canonical list is taken as-is, including the ``draft`` entries the registry
auto-created from values it could not place — ``Gemini-3-Flash(12`` and
``Seed-OSS-36B-Base(w`` are in it, truncated model names rather than
organizations. Filtering them out would make callers *louder* (an alias that
normalizes onto one is currently dropped as already-known), and this vocabulary
is only ever used to hold a warning back, so the untouched list is the
conservative one. It is also the honest picture of the registry to a maintainer
reading the diff.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from importlib import resources
from typing import Any, NamedTuple

#: Package-data file the snapshot lives in, under ``helpers/data/``. Reading it
#: goes through :func:`load_org_snapshot`, which uses ``importlib.resources`` so
#: an installed or zipped package works as well as a source checkout. Only the
#: refresh tool needs a filesystem path, and it derives one itself.
SNAPSHOT_NAME = 'org_registry.json'


class OrgVocabulary(NamedTuple):
    """Normalized lookups over one registry snapshot.

    Both are keyed by :func:`normalize_org_slug` output, so a lookup is
    insensitive to case and to ``-``/``_``/``.``/space punctuation. Normalizing
    is lossy — ``DeepAuto-AI`` and ``deepautoai`` are two canonical ids that
    collapse to one key — so ``identities`` is a set: the question it answers is
    "is this spelling already a name of record", and mapping the key back to one
    of the two ids would pick arbitrarily. ``load_org_snapshot`` still has the
    unnormalized ids for a caller that needs them.
    """

    identities: frozenset[str]
    second_names: dict[str, str]


def normalize_org_slug(value: str) -> str:
    """Collapse an organization slug to its punctuation-insensitive identity.

    ``moonshot-ai``, ``Moonshot AI`` and ``moonshotai`` all normalize alike:
    the registry's own ids are inconsistent about punctuation and case (it aims
    for HuggingFace-true casing, and HuggingFace is not consistent either), so
    treating those differences as meaningful would produce noise rather than
    signal.
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
