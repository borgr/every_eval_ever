"""The publisher resolver behind `check_developer_slug`.

`check_developer_slug`'s own tests (tests/test_validation_scope.py) cover the
warning. These cover the resolver underneath it, so a snapshot refresh that
changes what resolves fails here, next to the snapshot, rather than as a
puzzling change in validator output.
"""

from __future__ import annotations

import json

import pytest

from every_eval_ever.helpers import eval_card_registry


def test_a_second_name_resolves_to_the_canonical_publisher():
    for name, canonical in (
        ('mistral', 'mistralai'),
        ('zhipu-ai', 'zai'),
        ('glm', 'zai'),
        ('kimi', 'moonshotai'),
        ('granite', 'ibm'),
    ):
        assert eval_card_registry.second_name_of(name) == canonical, name


def test_a_publishers_own_names_are_not_second_names():
    """A canonical id, a recorded namespace, and variants of either."""
    for name in (
        'mistralai',
        'allenai',
        'meta-llama',
        'zai-org',
        'Anthropic',
        'moonshot-ai',
        'z-ai',
        # An alias that folds onto the canonical id is a punctuation variant of
        # it, not a different name: 'Mistral.AI' normalizes to 'mistralai'.
        'Mistral.AI',
    ):
        assert eval_card_registry.second_name_of(name) is None, name


def test_an_unknown_name_is_somebody_s_real_name():
    for name in ('mosaicml', 'aws-prototyping', 'unknown', '', '   '):
        assert eval_card_registry.second_name_of(name) is None, name


def test_a_non_string_is_not_a_publisher():
    for value in (None, 42, ['mistral'], {'id': 'mistral'}):
        assert eval_card_registry.second_name_of(value) is None


def test_one_publisher_s_second_name_cannot_shadow_another_s_real_name():
    """Every own name is collected before any alias resolves.

    `zai-org` is both Z.ai's namespace and one of its aliases, and `meta-llama`
    is both Meta's namespace and an alias. Resolving aliases first would report
    a record filed under the namespace the registry itself records.
    """
    snapshot = json.loads(
        eval_card_registry.SNAPSHOT_PATH.read_text(encoding='utf-8')
    )
    own_names = {
        eval_card_registry._normalize(name)
        for publisher in snapshot['publishers']
        for name in (publisher.get('id'), publisher.get('hf_org'))
        if isinstance(name, str) and name.strip()
    }

    aliases_that_are_also_own_names = {
        alias
        for publisher in snapshot['publishers']
        for alias in publisher.get('aliases') or []
        if eval_card_registry._normalize(alias) in own_names
    }

    assert aliases_that_are_also_own_names, (
        'the snapshot no longer exercises this case; if the registry stopped '
        'listing a namespace as an alias, this test is now vacuous'
    )
    for alias in aliases_that_are_also_own_names:
        assert eval_card_registry.second_name_of(alias) is None, alias


def test_the_snapshot_records_which_registry_commit_it_came_from():
    revision = eval_card_registry.snapshot_revision()

    assert revision is not None, (
        'refresh_publisher_snapshot.py records the registry revision; a '
        'snapshot without one was not written by it'
    )
    assert len(revision.removeprefix('').removesuffix('-dirty')) == 40


def test_the_snapshot_is_the_curated_seed_not_the_generated_one():
    """900 HF community namespaces carry no aliases and would only add noise."""
    snapshot = json.loads(
        eval_card_registry.SNAPSHOT_PATH.read_text(encoding='utf-8')
    )

    assert snapshot['_meta']['source'].endswith('seed/orgs.yaml')
    assert 50 <= snapshot['_meta']['count'] <= 300
    assert snapshot['_meta']['count'] == len(snapshot['publishers'])


@pytest.mark.parametrize('name', ['MISTRAL', ' mistral ', 'Mistral'])
def test_resolution_ignores_case_and_surrounding_space(name):
    assert eval_card_registry.second_name_of(name) == 'mistralai'
