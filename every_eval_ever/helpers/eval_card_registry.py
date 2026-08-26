"""Resolve a publisher spelling against the eval-card-registry's organizations.

The datastore is addressed by path, so one publisher writing under two names
occupies two directories and neither listing is complete. Deciding whether two
spellings are one publisher needs a vocabulary of second names, which is what
the registry's ``seed/orgs.yaml`` is: a canonical id per organization, the
namespace it publishes under, and the other names it appears as.

Read from a vendored snapshot (``publisher_snapshot.json``), so the validator
needs no registry install and no network. Refresh it with
``refresh_publisher_snapshot.py``.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

SNAPSHOT_PATH = Path(__file__).with_name('publisher_snapshot.json')

_NON_ALPHANUMERIC = re.compile(r'[^a-z0-9]+')


def _normalize(name: str) -> str:
    """Fold a spelling to what a datastore directory would not distinguish.

    Registry casing follows HuggingFace, which is not itself consistent
    ('anthropic' but 'Snowflake'), and punctuation in a slug is a house style
    rather than a different organization. So ``Anthropic``, ``anthropic``,
    ``moonshot-ai`` and ``moonshotai`` all fold together, and a case or
    punctuation variant of a canonical id is never reported as a second name.
    """
    return _NON_ALPHANUMERIC.sub('', name.strip().lower())


@lru_cache(maxsize=1)
def _publisher_index() -> tuple[dict[str, str], frozenset[str]]:
    """Map normalized second name -> canonical id, plus the names that are not one.

    ``own_names`` holds every canonical id and every recorded namespace. Those
    are a publisher's own names, so a record using one is filed where the
    registry says that publisher publishes, and there is nothing to warn about.
    They are collected across all organizations before any alias resolves, so
    one organization's alias cannot shadow another's real name.
    """
    snapshot = json.loads(SNAPSHOT_PATH.read_text(encoding='utf-8'))
    publishers = snapshot.get('publishers') or []

    own_names: set[str] = set()
    for publisher in publishers:
        for name in (publisher.get('id'), publisher.get('hf_org')):
            if isinstance(name, str) and name.strip():
                own_names.add(_normalize(name))

    second_names: dict[str, str] = {}
    for publisher in publishers:
        canonical = publisher.get('id')
        if not isinstance(canonical, str) or not canonical.strip():
            continue
        for alias in publisher.get('aliases') or []:
            if not isinstance(alias, str) or not alias.strip():
                continue
            key = _normalize(alias)
            if not key or key in own_names:
                continue
            second_names.setdefault(key, canonical)
    return second_names, frozenset(own_names)


def second_name_of(name: str) -> str | None:
    """Return the canonical id *name* is a second name for, or None.

    None covers four cases that are not evidence of a split directory: a
    canonical id, a namespace the registry records for a publisher, a case or
    punctuation variant of either, and a name the snapshot has never seen —
    which is somebody's real name as far as this repo can tell.
    """
    if not isinstance(name, str):
        return None
    key = _normalize(name)
    if not key:
        return None
    second_names, own_names = _publisher_index()
    if key in own_names:
        return None
    return second_names.get(key)


def snapshot_revision() -> str | None:
    """The registry commit this snapshot was taken from, when it recorded one."""
    snapshot = json.loads(SNAPSHOT_PATH.read_text(encoding='utf-8'))
    revision = (snapshot.get('_meta') or {}).get('registry_revision')
    return revision if isinstance(revision, str) else None
