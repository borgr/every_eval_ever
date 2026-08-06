"""Tests for the shared eval-card-registry vocabulary and resolver.

Two layers, tested separately: what the refresh tool *derives* from the
registry's list endpoints (pure functions over raw records), and what the
bundled snapshot then answers (the vocabulary consumers actually read).
"""

import pytest

from every_eval_ever.helpers.eval_card_registry import (
    Registry,
    gaps,
    iter_org_identities,
    normalize,
    second_name_of,
    snapshot_meta,
)
from every_eval_ever.tools.refresh_eval_card_registry import (
    org_identities,
    org_second_names,
)


def _org(org_id, hf_org=None, review_status='reviewed'):
    return {'id': org_id, 'hf_org': hf_org, 'review_status': review_status}


def _alias(raw_value, canonical_id, status='confirmed'):
    return {
        'raw_value': raw_value,
        'canonical_id': canonical_id,
        'status': status,
    }


# ---------------------------------------------------------------------------
# Deriving the vocabulary
# ---------------------------------------------------------------------------


def test_a_recorded_namespace_is_an_identity_not_an_alias():
    """``meta-llama`` is Meta, so it must not read as a second name."""
    identities = org_identities([_org('meta', 'meta-llama'), _org('cohere')])

    assert identities['metallama'] == 'meta'
    assert identities['meta'] == 'meta'
    assert identities['cohere'] == 'cohere'


def test_a_spelling_two_organizations_answer_to_names_neither():
    """``DeepAuto-AI`` and ``deepautoai`` are both canonical ids.

    Awarding the shared spelling to one makes the other resolve to an
    organization that is not itself, and this mapping decides a published
    ``model_info.developer``.
    """
    orgs = [_org('DeepAuto-AI'), _org('deepautoai')]

    assert org_identities(orgs) == org_identities(orgs[::-1]) == {}


def test_a_namespace_does_not_claim_a_contested_spelling():
    identities = org_identities(
        [_org('DeepAuto-AI'), _org('deepautoai'), _org('other', 'deepauto.ai')]
    )

    assert 'deepautoai' not in identities


def test_second_names_keep_only_a_genuinely_different_name():
    identities = org_identities(
        [_org('mistralai'), _org('meta', 'meta-llama'), _org('zai')]
    )
    second_names = org_second_names(
        [
            _alias('Mistral', 'mistralai'),
            # Restates an identity: a canonical id, then a namespace.
            _alias('Mistral AI', 'mistralai'),
            _alias('meta-llama', 'meta'),
            # Unconfirmed, and pointing at an organization not in the list.
            _alias('Mistral Large', 'mistralai', status='pending'),
            _alias('Kimi', 'moonshotai'),
            # One normalized spelling claimed by two organizations.
            _alias('GLM', 'zai'),
            _alias('glm', 'meta'),
        ],
        identities,
    )

    assert second_names == {'mistral': 'mistralai'}


def test_second_names_yield_to_a_canonical_id_of_another_organization():
    """The registry sometimes has two canonical ids for one publisher.

    ``AI21 Labs`` is a confirmed alias of ``ai21`` while ``ai21-labs`` is its
    own canonical id. Answering with either would assert an ordering between
    them that the registry has not made, so the alias is dropped.
    """
    identities = org_identities([_org('ai21'), _org('ai21-labs')])

    assert org_second_names([_alias('AI21 Labs', 'ai21')], identities) == {}


# ---------------------------------------------------------------------------
# Reading the bundled snapshot
# ---------------------------------------------------------------------------


def test_normalize_collapses_case_and_punctuation():
    assert (
        normalize('Moonshot AI')
        == normalize('moonshot-ai')
        == normalize('moonshot_ai')
        == 'moonshotai'
    )
    assert normalize('  Z.AI  ') == 'zai'
    assert normalize('Win Rate') == normalize('win_rate') == 'winrate'


def test_second_name_of_answers_only_for_a_different_name():
    assert second_name_of('mistral') == 'mistralai'
    assert second_name_of('AI2') == 'allenai'
    # A canonical id, and a HuggingFace namespace the registry records for one.
    assert second_name_of('mistralai') is None
    assert second_name_of('meta-llama') is None
    # Unknown to the registry, and not a string at all.
    assert second_name_of('mosaicml') is None
    assert second_name_of(None) is None


def test_second_names_and_identities_never_overlap():
    """A spelling cannot be both a name of record and a second name for one."""
    identities = {key for key, _ in iter_org_identities()}

    assert identities
    assert not [key for key in identities if second_name_of(key)]


def test_org_resolution_reads_namespaces_and_second_names_alike():
    """A converter asks "which organization is this" and gets one answer."""
    registry = Registry()

    assert registry.org('meta-llama').canonical_id == 'meta'
    assert registry.org('meta-llama').strategy == 'snapshot'
    assert registry.org('AI2').canonical_id == 'allenai'
    assert registry.org('AI2').strategy == 'snapshot_alias'
    assert registry.org('meta-llama').reviewed


def test_a_canonical_id_always_resolves_to_itself():
    """Including the four whose normalized spelling another id also answers to.

    Those spellings are unowned in the snapshot, so an id like ``deepautoai``
    reaches itself only by name. Resolving it to its punctuation twin would
    publish one organization's records under another's directory.
    """
    registry = Registry()
    for org_id in ('deepautoai', 'DeepAuto-AI', 'mistralai', 'meta', 'zai'):
        resolution = registry.org(org_id)
        assert resolution.canonical_id == org_id, org_id

    assert registry.org('deepautoai').strategy == 'snapshot_exact'
    assert second_name_of('deepautoai') is None


def test_snapshot_records_its_own_provenance():
    """A vendored snapshot without provenance cannot be audited."""
    meta = snapshot_meta()

    assert 'read-only' in meta['source']
    assert meta['retrieved_date']
    assert meta['counts']['orgs'] > 0
    assert 'refresh_eval_card_registry' in meta['note']
    # The gaps are recorded rather than silently absent, so a consumer can
    # report them and a refresh can be diffed.
    assert 'metric:avg_length' in gaps()


# ---------------------------------------------------------------------------
# The opt-in live path
# ---------------------------------------------------------------------------


def test_registry_never_resolves_in_a_mode_that_creates_canonicals():
    """The live path must not write to a shared registry.

    ``POST /resolve`` defaults to ``mode="resolve"``, which auto-creates a draft
    canonical for anything it cannot place. Only ``mode="exact"`` is
    side-effect-free, so that is the only mode this module may send.
    """
    sent = {}

    class _Response:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {'canonical_id': None, 'created_new': False}

    def _post(url, json=None, timeout=None):
        sent.update(json or {})
        return _Response()

    registry = Registry(live=True)
    module = pytest.importorskip('requests')
    original = module.post
    module.post = _post
    try:
        registry.metric('a_column_the_registry_has_never_heard_of')
    finally:
        module.post = original

    assert sent['mode'] == 'exact'


def test_live_registry_failure_is_never_fatal():
    """A registry outage degrades provenance; it does not fail a conversion."""

    def _post(url, json=None, timeout=None):
        raise OSError('registry unreachable')

    registry = Registry(live=True)
    module = pytest.importorskip('requests')
    original = module.post
    module.post = _post
    try:
        resolution = registry.metric('a_column_with_no_canonical')
    finally:
        module.post = original

    assert resolution.canonical_id is None
    assert resolution.strategy == 'registry_unavailable'
    assert 'registry unreachable' in registry.live_error


def test_one_outage_does_not_relabel_every_later_miss():
    """A clean miss and an outage are different facts about a record.

    ``live_error`` is a run-level aggregate that the report needs; deciding a
    single resolution's strategy from it made every lookup after the first fault
    claim the registry was unavailable when it had answered.
    """
    calls = []

    class _Response:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {'canonical_id': None, 'created_new': False}

    def _post(url, json=None, timeout=None):
        calls.append(json['raw_value'])
        if len(calls) == 1:
            raise OSError('registry unreachable')
        return _Response()

    registry = Registry(live=True)
    module = pytest.importorskip('requests')
    original = module.post
    module.post = _post
    try:
        first = registry.metric('a_column_the_registry_never_answers_for')
        second = registry.metric('another_column_with_no_canonical')
    finally:
        module.post = original

    assert first.strategy == 'registry_unavailable'
    assert second.strategy == 'no_canonical'
    # The outage is still reported once, for the run as a whole.
    assert 'registry unreachable' in registry.live_error
