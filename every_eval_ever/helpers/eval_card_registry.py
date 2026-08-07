"""Canonical ids from the eval-card-registry, resolved offline.

The registry (``https://evaleval-entity-registry.hf.space``) is the shared
canonicalization service for EEE, and this module is how anything in this repo
reads it — converters deciding what to publish, and the validator deciding what
to warn about. It answers four questions a source must not answer by hand:

- ``model_info.developer`` — the canonical **organization** id. The registry
  distinguishes the organization from the HuggingFace namespace its models are
  published under (``meta`` and ``meta-llama``, ``alibaba`` and ``qwen``,
  ``zai`` and ``zai-org``), and both are real identities for the same
  organization rather than drift. So ``model_info.id`` keeps the namespace,
  because that is the repo id that resolves, and ``developer`` carries the
  canonical org id.
- whether a publisher name is a **second name** for an organization already in
  the registry (``Mistral`` for ``mistralai``, ``AI2`` for ``allenai``) rather
  than an identity of its own. The datastore gives each publisher one directory
  (``data/<collection>/<publisher>/``, taken from the ``model_info.id``
  namespace and from ``developer`` for an id without one — see
  :func:`every_eval_ever.helpers.io.datastore_path_components`), so one
  publisher under two names is two directories and neither listing is complete
  — see :func:`second_name_of`.
- the **metric** id and the score bounds that come with it. The registry's
  ``win-rate`` is declared on ``[0, 100]``, which is the scale the AlpacaEval
  leaderboard CSV already publishes, so this settles a question two prior
  implementations of that converter answered differently.
- the **benchmark** and **harness** ids, for the evaluation name and the
  library that produced it.

Resolution reads a vendored snapshot of the registry's read-only list
endpoints (:data:`SNAPSHOT_PATH`), not its ``/resolve`` endpoint. That is a
deliberate inversion of the usual "resolve live" advice, for a reason that is
easy to verify: ``POST /api/v1/resolve`` defaults to ``mode="resolve"``, which
**auto-creates a draft canonical** for any value it cannot place. Resolving 226
leaderboard rows that way would add hundreds of draft models to a shared
registry as a side effect of a read-only conversion, and the registry is already
18157 draft models to 5298 reviewed ones. A validator calling it would be worse:
checking a file would write to a shared registry. Reading a snapshot is
deterministic, needs no network, and cannot write.

``--registry-live`` opts into a live check anyway, and uses ``mode="exact"``,
which is the one mode that resolves without creating anything. It is never
fatal: a live failure falls back to the snapshot, and an unresolved value falls
back to the source-derived spelling marked unverified, so a registry outage
degrades provenance instead of blocking a conversion.

That last property is deliberate, and it is the half of a two-part contract:
**conversion is soft, validation is hard.** A converter records what the
registry said, including that it said nothing, and finishes; the validator is
where an unresolved id is refused. Putting the refusal in the converter instead
would mean a registry outage — or a model the registry has not been taught yet
— destroys a day's extraction rather than marking it, and marking it is enough
because the record carries ``strategy`` and can be re-resolved from disk. A
converter that fails closed on a shared network service also cannot be run
reproducibly against an archived leaderboard, which is most of the point.

Refresh the snapshot with
``python -m every_eval_ever.tools.refresh_eval_card_registry``.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Tuple

REGISTRY_BASE_URL = 'https://evaleval-entity-registry.hf.space'

SNAPSHOT_NAME = 'eval_card_registry.json'

#: Where :mod:`every_eval_ever.tools.refresh_eval_card_registry` writes the
#: snapshot in a source checkout. Reading goes through :func:`load_snapshot` so
#: an installed or zipped package works too.
SNAPSHOT_PATH = Path(__file__).resolve().parent / 'data' / SNAPSHOT_NAME

#: Only this mode resolves without creating a draft canonical.
_SIDE_EFFECT_FREE_MODE = 'exact'

_LIVE_TIMEOUT = 30

#: Snapshot section -> registry entity type. Spelled out because the entity type
#: is not the section name minus an ``s`` (``harnesses`` -> ``harness``).
_SECTION_ENTITY_TYPES = {
    'metrics': 'metric',
    'benchmarks': 'benchmark',
    'harnesses': 'harness',
}


def normalize(value: str) -> str:
    """Collapse a name to its punctuation-insensitive identity.

    ``win_rate``, ``Win Rate`` and ``win-rate`` normalize alike, which is what
    lets a leaderboard column name find a canonical id; so do ``moonshot-ai``,
    ``Moonshot AI`` and ``moonshotai``. The registry's own ids are inconsistent
    about punctuation and case (it aims for HuggingFace-true casing, and
    HuggingFace is not consistent either), so treating those differences as
    meaningful would produce noise rather than signal.

    Discarding punctuation is a weaker claim than folding case, so
    :meth:`Registry.org` tries the case-folded spelling first and labels the two
    outcomes differently rather than treating this as the only key.
    """
    if not isinstance(value, str):
        return ''
    return re.sub(r'[^a-z0-9]+', '', value.strip().lower())


def load_snapshot() -> Dict[str, Any]:
    """Return the bundled registry snapshot as parsed JSON."""
    resource = resources.files('every_eval_ever.helpers').joinpath(
        'data', SNAPSHOT_NAME
    )
    return json.loads(resource.read_text(encoding='utf-8'))


@lru_cache(maxsize=1)
def _snapshot() -> Dict[str, Any]:
    return load_snapshot()


def snapshot_meta() -> Dict[str, Any]:
    """Provenance of the vendored snapshot: endpoints, date, counts."""
    return dict(_snapshot().get('_meta') or {})


class Resolution(NamedTuple):
    """One canonical id, and how much weight it deserves.

    ``strategy`` and ``review_status`` are published alongside the value they
    produced: a caller reading a record can tell a ``reviewed`` canonical from a
    ``draft`` one, and either from a source-derived fallback the registry has
    never seen.
    """

    raw_value: str
    entity_type: str
    canonical_id: Optional[str]
    review_status: Optional[str]
    #: ``snapshot_exact`` | ``snapshot_identifier``
    #: | ``snapshot_alias_identifier`` | ``snapshot_normalized``
    #: | ``snapshot_alias_normalized`` | ``snapshot`` | ``live_exact``
    #: | ``no_canonical`` | ``registry_disabled`` | ``registry_unavailable``.
    #: The org tiers are ordered by how much of the spelling was discarded to
    #: reach a match — see :meth:`Registry.org`.
    strategy: str
    record: Dict[str, Any] = {}

    @property
    def resolved(self) -> bool:
        return self.canonical_id is not None

    @property
    def reviewed(self) -> bool:
        """True when a human has vetted this entry in the registry."""
        return self.review_status == 'reviewed'

    def provenance(self, prefix: str) -> Dict[str, Optional[str]]:
        """Fields to publish in ``additional_details`` for this resolution."""
        return {
            f'{prefix}_registry_id': self.canonical_id,
            f'{prefix}_registry_strategy': self.strategy,
            f'{prefix}_registry_review_status': self.review_status,
        }


def _unresolved(raw_value: str, entity_type: str, strategy: str) -> Resolution:
    return Resolution(
        raw_value=raw_value,
        entity_type=entity_type,
        canonical_id=None,
        review_status=None,
        strategy=strategy,
    )


class Registry:
    """Offline resolver over the vendored snapshot, with an opt-in live check.

    Args:
        enabled: When False every lookup returns unresolved with strategy
            ``registry_disabled``, so ``--no-registry-resolve`` produces records
            that are explicit about having had no registry opinion rather than
            records that quietly look source-derived.
        live: Additionally consult ``POST /api/v1/resolve/batch`` in
            ``mode="exact"`` for values the snapshot cannot place. Never fatal.
        base_url: Registry base URL, for tests and staging deployments.
    """

    def __init__(
        self,
        enabled: bool = True,
        live: bool = False,
        base_url: str = REGISTRY_BASE_URL,
    ) -> None:
        self.enabled = enabled
        self.live = live and enabled
        self.base_url = base_url
        #: Values the live endpoint was asked about, so a run can report it.
        self.live_queries = 0
        self.live_hits = 0
        self.live_error: Optional[str] = None
        self._live_cache: Dict[
            Tuple[str, str], Tuple[Optional[Dict[str, Any]], Optional[str]]
        ] = {}

    # -- organizations ----------------------------------------------------

    def org(self, slug: str) -> Resolution:
        """Resolve a HuggingFace namespace or org name to a canonical org id.

        Both a canonical id and a namespace the registry records for one resolve
        here, because both name the same organization.

        Five tiers, tried in order of how much of the spelling had to be thrown
        away to get a match, and ``strategy`` records which one answered:

        ``snapshot_exact``
            The value **is** a canonical id. Tried first because two ids can
            collapse to one normalized spelling (``DeepAuto-AI`` and
            ``deepautoai``) and that spelling is left unowned rather than
            awarded to one of them, so without this an id would resolve to a
            different organization or not at all.
        ``snapshot_identifier`` / ``snapshot_alias_identifier``
            A spelling the registry records — a canonical id, a ``hf_org``
            namespace, or a confirmed alias — matched case-insensitively.
        ``snapshot_normalized`` / ``snapshot_alias_normalized``
            The same, but only after punctuation was discarded as well.

        The split matters because the two are different claims. ``meta-llama``
        is a namespace Meta declares; ``metallama`` is a spelling nobody
        declared that merely collapses onto one. Guessing from punctuation is
        how ``anthropic/claude-2.1`` becomes ``claude-21``, so a reader of a
        published record is entitled to know which happened. Today no
        organization on either AlpacaEval leaderboard reaches its canonical id
        by the normalized tiers; every one of them matches a recorded
        identifier.
        """
        if not self.enabled:
            return _unresolved(slug, 'org', 'registry_disabled')
        snapshot = _snapshot()
        exact = slug.strip() if isinstance(slug, str) else slug
        if exact in snapshot['org_review_status']:
            return Resolution(
                raw_value=slug,
                entity_type='org',
                canonical_id=exact,
                review_status=snapshot['org_review_status'][exact],
                strategy='snapshot_exact',
            )
        # Sections a snapshot taken before the tiers were split does not carry;
        # such a snapshot resolves by the normalized tiers alone, as it did.
        folded = exact.lower() if isinstance(exact, str) else ''
        key = normalize(slug)
        for section, lookup, strategy in (
            ('org_identity_spellings', folded, 'snapshot_identifier'),
            ('org_alias_spellings', folded, 'snapshot_alias_identifier'),
            ('org_identities', key, 'snapshot_normalized'),
            ('org_aliases', key, 'snapshot_alias_normalized'),
        ):
            canonical = (snapshot.get(section) or {}).get(lookup)
            if canonical is not None:
                return Resolution(
                    raw_value=slug,
                    entity_type='org',
                    canonical_id=canonical,
                    review_status=snapshot['org_review_status'].get(canonical),
                    strategy=strategy,
                )
        return self._live('org', slug)

    # -- metrics / benchmarks / harnesses ---------------------------------

    def metric(self, name: str) -> Resolution:
        """Resolve a leaderboard column name to a canonical metric entry.

        The entry carries ``min_score``/``max_score``/``lower_is_better``, which
        is the point: the registry, not this adapter, decides the scale a score
        is published on.
        """
        return self._keyed('metrics', 'metric', name)

    def benchmark(self, name: str) -> Resolution:
        return self._keyed('benchmarks', 'benchmark', name)

    def harness(self, name: str) -> Resolution:
        return self._keyed('harnesses', 'harness', name)

    def _keyed(self, section: str, entity_type: str, name: str) -> Resolution:
        """Look up a value the snapshot stores under its query spelling.

        The snapshot records a ``None`` entry for a query the registry has no
        canonical for, which is a different fact from "not in the snapshot": the
        first is a known gap this adapter has already reported, the second means
        the snapshot needs a refresh. Both fall back, only the second tries the
        network.
        """
        if not self.enabled:
            return _unresolved(name, entity_type, 'registry_disabled')
        entries = _snapshot()[section]
        if name not in entries:
            return self._live(entity_type, name)
        entry = entries[name]
        if entry is None:
            return _unresolved(name, entity_type, 'no_canonical')
        return Resolution(
            raw_value=name,
            entity_type=entity_type,
            canonical_id=entry['id'],
            review_status=entry.get('review_status'),
            strategy='snapshot',
            record=entry,
        )

    # -- opt-in live path -------------------------------------------------

    def _live(self, entity_type: str, raw_value: str) -> Resolution:
        """Ask the registry directly, in the mode that creates nothing."""
        if not self.live:
            return _unresolved(raw_value, entity_type, 'no_canonical')
        payload, error = self._live_lookup(entity_type, raw_value)
        if payload is None:
            # This lookup's own outcome, not the run's: ``live_error`` is a
            # sticky aggregate, and reading it here would relabel every later
            # clean miss as an outage in the published record.
            strategy = 'registry_unavailable' if error else 'no_canonical'
            return _unresolved(raw_value, entity_type, strategy)
        return Resolution(
            raw_value=raw_value,
            entity_type=entity_type,
            canonical_id=payload['canonical_id'],
            review_status=payload.get('review_status'),
            strategy='live_exact',
            record=payload,
        )

    def _live_lookup(
        self, entity_type: str, raw_value: str
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Return ``(payload, error)`` for one lookup — a hit, a miss, or a fault.

        A miss and a fault are both ``payload is None`` and mean different things
        to a reader of the record, so they are cached and reported separately.
        """
        cache_key = (entity_type, raw_value)
        if cache_key in self._live_cache:
            return self._live_cache[cache_key]
        result = None
        error_text = None
        try:
            import requests

            self.live_queries += 1
            response = requests.post(
                f'{self.base_url}/api/v1/resolve',
                json={
                    'raw_value': raw_value,
                    'entity_type': entity_type,
                    'mode': _SIDE_EFFECT_FREE_MODE,
                    'source_config': 'alpaca_eval',
                },
                timeout=_LIVE_TIMEOUT,
            )
            response.raise_for_status()
            payload = response.json()
            # Belt and braces: `mode="exact"` must never create a canonical, so
            # a response claiming it did is a contract change, not a resolution.
            if payload.get('canonical_id') and not payload.get('created_new'):
                result = payload
                self.live_hits += 1
        except Exception as error:  # never fatal — provenance, not data
            error_text = f'{type(error).__name__}: {error}'
            self.live_error = error_text
        self._live_cache[cache_key] = (result, error_text)
        return result, error_text

    # -- reporting --------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        """A summary of what the registry contributed, for the run report."""
        meta = snapshot_meta()
        return {
            'enabled': self.enabled,
            'live': self.live,
            'snapshot_date': meta.get('retrieved_date'),
            'snapshot_counts': meta.get('counts'),
            'live_queries': self.live_queries,
            'live_hits': self.live_hits,
            'live_error': self.live_error,
        }


def snapshot_gaps(snapshot: Dict[str, Any]) -> List[str]:
    """``entity_type:query`` for every query *snapshot* has no canonical for."""
    return [
        f'{entity_type}:{query}'
        for section, entity_type in _SECTION_ENTITY_TYPES.items()
        for query, entry in snapshot[section].items()
        if entry is None
    ]


def gaps() -> List[str]:
    """Queries the vendored snapshot records as having no canonical entry.

    These are the registry-side follow-ups a conversion cannot fix on its own:
    minting a canonical is a lasting namespace decision and belongs in a PR to
    the registry, not in an adapter.
    """
    return snapshot_gaps(_snapshot())


def iter_org_identities() -> Iterable[Tuple[str, str]]:
    """(normalized spelling, canonical org id) pairs, for tests and tooling."""
    return _snapshot()['org_identities'].items()


def second_name_of(slug: str) -> Optional[str]:
    """Return the canonical org id when ``slug`` is a *second name* for it.

    A second name is a confirmed alias that is a genuinely **different** name
    for an organization the registry already knows: ``Mistral`` for
    ``mistralai``, ``AI2`` for ``allenai``, or a model family such as ``glm``
    used where its publisher belongs. Two names for one publisher split it
    across two datastore directories, so a caller that groups by publisher —
    or warns a contributor about to create the second one — wants to know.

    ``None`` in the three cases where a name is *not* evidence of a split: a
    canonical id, a HuggingFace namespace the registry records for one
    (``meta-llama`` is Meta), and a spelling the registry has never seen — no
    opinion rather than a guess.

    This is the read-side counterpart of :meth:`Registry.org`, which answers
    "what should this be called" for a converter. Here the question is only
    "is this a second name", so an identity gets ``None`` rather than itself.
    """
    if not isinstance(slug, str):
        return None
    key = normalize(slug)
    if not key:
        return None
    snapshot = _snapshot()
    # The snapshot builder already drops an alias that normalizes onto an
    # identity, so this check is redundant against a snapshot it wrote. It is
    # kept because the guarantee belongs to this function: a caller uses the
    # answer to warn a contributor, and a hand-edited or older snapshot must
    # not be able to turn a canonical id into a "second name".
    if slug.strip() in snapshot['org_review_status']:
        return None
    if key in snapshot['org_identities']:
        return None
    return snapshot['org_aliases'].get(key)
