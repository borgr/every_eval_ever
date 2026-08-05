"""Tests for the bundled eval-card-registry organization vocabulary."""

from every_eval_ever.helpers.org_registry import (
    load_org_snapshot,
    normalize_org_slug,
    org_vocabulary,
    second_name_of,
)
from every_eval_ever.tools.refresh_org_registry import (
    build_snapshot,
    render_snapshot,
)


def test_snapshot_has_the_three_groups_the_loader_expects():
    snapshot = load_org_snapshot()
    canonical_ids = set(snapshot['orgs'])

    assert isinstance(snapshot['orgs'], list)
    assert all(isinstance(org_id, str) for org_id in snapshot['orgs'])
    for group in ('hf_orgs', 'second_names'):
        assert all(
            isinstance(key, str) and value in canonical_ids
            for key, value in snapshot[group].items()
        ), group
    # The snapshot is refreshed from a live service, so it says where from.
    assert snapshot['_source'].startswith('https://')
    assert 'refresh_org_registry' in snapshot['_refresh']


def test_normalize_collapses_case_and_punctuation():
    assert (
        normalize_org_slug('Moonshot AI')
        == normalize_org_slug('moonshot-ai')
        == normalize_org_slug('moonshot_ai')
        == 'moonshotai'
    )
    assert normalize_org_slug('  Z.AI  ') == 'zai'


def test_second_names_and_identities_never_overlap():
    vocabulary = org_vocabulary()

    assert vocabulary.identities
    assert vocabulary.second_names
    assert not vocabulary.identities & set(vocabulary.second_names)


def test_second_name_of_answers_only_for_a_different_name():
    assert second_name_of('mistral') == 'mistralai'
    assert second_name_of('AI2') == 'allenai'
    # A canonical id, and a HuggingFace namespace the registry records for one.
    assert second_name_of('mistralai') is None
    assert second_name_of('meta-llama') is None
    # Unknown to the registry, and not a string at all.
    assert second_name_of('mosaicml') is None
    assert second_name_of(None) is None


def _org(org_id, hf_org=None):
    return {'id': org_id, 'hf_org': hf_org}


def _alias(raw_value, canonical_id, status='confirmed'):
    return {
        'raw_value': raw_value,
        'canonical_id': canonical_id,
        'status': status,
    }


def test_build_snapshot_keeps_only_namespaces_that_add_a_spelling():
    snapshot = build_snapshot(
        [_org('meta', 'meta-llama'), _org('anthropic', 'anthropic')],
        [],
    )

    assert snapshot['orgs'] == ['anthropic', 'meta']
    # 'anthropic' would normalize onto its own canonical id, so it is dropped.
    assert snapshot['hf_orgs'] == {'meta-llama': 'meta'}


def test_build_snapshot_drops_aliases_that_add_nothing_or_conflict():
    snapshot = build_snapshot(
        [_org('mistralai'), _org('meta', 'meta-llama'), _org('zai')],
        [
            _alias('Mistral', 'mistralai'),
            # Already an identity: a canonical id, then a namespace.
            _alias('Mistral AI', 'mistralai'),
            _alias('meta-llama', 'meta'),
            # Unconfirmed, and pointing at an organization not in the list.
            _alias('Mistral Large', 'mistralai', status='pending'),
            _alias('Kimi', 'moonshotai'),
            # One normalized spelling claimed by two organizations.
            _alias('GLM', 'zai'),
            _alias('glm', 'meta'),
        ],
    )

    assert snapshot['second_names'] == {'Mistral': 'mistralai'}


def test_build_snapshot_yields_to_a_canonical_id_of_another_organization():
    """The registry sometimes has two canonical ids for one publisher.

    ``AI21 Labs`` is a confirmed alias of ``ai21`` while ``ai21-labs`` is its
    own canonical id. Answering with either would assert an ordering between
    them that the registry has not made, so the alias is dropped.
    """
    snapshot = build_snapshot(
        [_org('ai21'), _org('ai21-labs')],
        [_alias('AI21 Labs', 'ai21')],
    )

    assert snapshot['second_names'] == {}


def test_render_snapshot_is_stable_so_refreshes_diff_cleanly():
    orgs = [_org('zai', 'zai-org'), _org('mistralai')]
    aliases = [_alias('THUDM', 'zai'), _alias('Mistral', 'mistralai')]

    rendered = render_snapshot(build_snapshot(orgs, aliases))

    # Endpoint order is not guaranteed, so the file must not depend on it.
    assert rendered == render_snapshot(
        build_snapshot(orgs[::-1], aliases[::-1])
    )
    assert rendered.endswith('\n')
