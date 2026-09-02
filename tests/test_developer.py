"""`get_developer` must answer with the publishing namespace, not the company.

The datastore's developer folder is `model_info.id`'s prefix
(`datastore-gate.md` §path), so a bare model name has to resolve to the same
namespace its slashed id would carry. When it folded to the parent company
instead, one model reached two directories depending on how a source happened to
spell it, which is #272 and a meaningful share of the 996 publisher directories
in the published datastore.
"""

from __future__ import annotations

import pytest

from every_eval_ever.helpers.developer import (
    DEVELOPER_PATTERNS,
    HEAD_ONLY_PATTERNS,
    get_developer,
)

#: (slashed id, bare name) for one model. Every namespace here is HuggingFace's
#: own spelling, taken from its organization record rather than inferred.
SAME_MODEL_TWO_SPELLINGS = [
    ('Qwen/Qwen3-32B', 'qwen3-32b'),
    ('meta-llama/Llama-3.1-8B-Instruct', 'llama-3.1-8b-instruct'),
    ('facebook/opt-1.3b', 'opt-1.3b'),
    ('mistralai/Mistral-Large-2411', 'mistral-large-2411'),
    ('deepseek-ai/DeepSeek-V3', 'deepseek-v3'),
    ('zai-org/GLM-4.6', 'glm-4.6'),
    ('EleutherAI/pythia-1b', 'pythia-1b'),
    ('CohereForAI/c4ai-command-r-plus', 'command-r-plus'),
    ('ai21labs/AI21-Jamba-1.5-Large', 'jamba-1.5-large'),
    ('Snowflake/snowflake-arctic-instruct', 'arctic-instruct'),
    ('togethercomputer/RedPajama-INCITE-7B-Base', 'redpajama-incite-7b-base'),
    ('allenai/OLMo-2-1124-7B', 'olmo-2-1124-7b'),
    ('microsoft/phi-4', 'phi-4'),
    ('google/gemma-3-27b-it', 'gemma-3-27b-it'),
    ('facebook/bart-large', 'bart-large'),
    ('microsoft/deberta-v3-large', 'deberta-v3-large'),
    ('nvidia/Nemotron-H-8B-Base-8K', 'nemotron-h-8b-base-8k'),
    ('distilbert/distilbert-base-uncased', 'distilbert-base-uncased'),
    ('swiss-ai/Apertus-70B-Instruct-2509', 'apertus-70b-instruct-2509'),
    ('rednote-hilab/dots.llm1.inst', 'dots.llm1.inst'),
    ('LGAI-EXAONE/EXAONE-4.0-32B', 'exaone-4.0-32b'),
    ('meituan-longcat/LongCat-Flash-Chat', 'longcat-flash-chat'),
]


@pytest.mark.parametrize('slashed,bare', SAME_MODEL_TWO_SPELLINGS)
def test_one_model_gets_one_developer_however_it_is_spelled(
    slashed: str, bare: str
) -> None:
    assert get_developer(bare) == get_developer(slashed) == slashed.split('/')[0]


def test_a_closed_model_keeps_its_company_because_it_has_no_namespace():
    """A model with no HF repo is addressed `{org}/{slug}`, so the org is right."""
    for bare, developer in (
        ('gpt-4.1-mini', 'openai'),
        ('claude-opus-4-5', 'anthropic'),
        ('grok-4', 'xai'),
        ('nova-pro', 'amazon'),
    ):
        assert get_developer(bare) == developer


def test_every_pattern_answers_with_one_path_component():
    """A value with a slash would be flattened into a different directory."""
    for table in (DEVELOPER_PATTERNS, HEAD_ONLY_PATTERNS):
        for pattern, developer in table.items():
            assert '/' not in developer, pattern
            assert developer == developer.strip(), pattern
            assert developer, pattern


def test_a_head_only_family_does_not_claim_a_name_that_merely_contains_it():
    """The families most often embedded in someone else's system name.

    Each left-hand name is a third party's model or method that carries the
    family as a suffix. Answering with the family's namespace would file
    another group's score under it, which is worse than dropping the row.
    """
    for borrowed, owned in (
        ('BIGBIRD-RoBERTa', 'RoBERTa'),
        ('KnowRL-Nemotron-1.5B', 'Nemotron-4 15B'),
        ('B3D-RWKV-7.2B', 'RWKV-4 14B'),
        ('K-EXAONE 2.0', 'EXAONE 4.0 32B'),
        ('AudioLDM-S-Full-RoBERTa', 'RoBERTa'),
        ('LION-Mamba-L', 'Nemotron 3 Super'),
    ):
        assert get_developer(borrowed) == 'unknown', borrowed
        assert get_developer(owned) != 'unknown', owned


def test_the_two_pattern_tables_do_not_disagree_about_a_family():
    overlap = set(DEVELOPER_PATTERNS) & set(HEAD_ONLY_PATTERNS)
    assert not overlap, overlap


def test_a_family_that_runs_on_into_a_word_is_not_that_family():
    """Several family names are also word fragments.

    Each of these is another group's system that merely starts with the
    letters, so answering with the family's namespace would file its score
    under OpenAI, Meta, Alibaba, MosaicML or 01-ai.
    """
    for name in (
        'AdaFace R100 (WebFace4M)',
        'AdapterTune (ViT-B/16)',
        'AdaCLIP (zero-shot)',
        'OptiMer',
        'LlamaGen-3B',
        'QwenLong-L1.5-30B-A3B',
        'MPTSNet',
        'YingLong 300M (zero-shot)',
        'MistralLite',
        'GPTQ',
    ):
        assert get_developer(name) == 'unknown', name


def test_a_bounded_family_still_matches_its_own_spellings():
    for name, developer in (
        ('ada', 'openai'),
        ('text-ada-001', 'openai'),
        ('opt-1.3b', 'facebook'),
        ('Yi-34B', '01-ai'),
        ('Qwen3-32B', 'Qwen'),
        ('Phi-4', 'microsoft'),
        ('o3-mini', 'openai'),
        ('Command R+', 'CohereForAI'),
        ('Nova Pro', 'amazon'),
    ):
        assert get_developer(name) == developer, name


def test_a_family_whose_own_releases_run_on_keeps_matching_them():
    """`OLMoE` and `T5Gemma` are the publisher's own names, not a third
    party's, so they are deliberately outside `BOUNDED_PATTERNS`."""
    for name, developer in (
        ('T5Gemma 9B-9B PrefixLM', 'google'),
        ('OLMoE-1B-7B (5-shot)', 'allenai'),
        ('olmOCR2', 'allenai'),
        ('KimiVL-16B-A3B-Think', 'moonshotai'),
        ('HunyuanOCR-1.5 (1B)', 'tencent'),
    ):
        assert get_developer(name) == developer, name


def test_a_slash_that_is_not_a_repo_id_does_not_become_a_developer():
    """Leaderboards write a patch size, a variant and a setting with a slash.

    Splitting on the slash regardless invented the publishers `ViT-L`,
    `SR-DiT-B` and `Claude Opus 5 (w`, and filed a known model's score under
    each of them.
    """
    assert get_developer('ViT-L/16') == 'unknown'
    assert get_developer('CapPa L/14') == 'unknown'
    assert get_developer('SR-DiT-B/1') == 'unknown'
    assert get_developer('3B/1B MoE') == 'unknown'
    assert get_developer('Claude Opus 5 (w/ tools)') == 'anthropic'
    assert get_developer('GPT-5.2 (w/ Python)') == 'openai'


def test_a_repo_id_still_answers_with_its_namespace():
    assert get_developer('meta-llama/Llama-3-8B') == 'meta-llama'
    assert get_developer('facebook/opt-1.3b') == 'facebook'
    assert get_developer('AutoArk-AI/ARK-ASR-0.6B') == 'AutoArk-AI'
    assert get_developer('rednote-hilab/dots.llm1.inst') == 'rednote-hilab'
    assert get_developer('org/family/model:revision') == 'org'


def test_an_unknown_model_is_not_guessed_at():
    assert get_developer('some-model-nobody-registered') == 'unknown'
    assert get_developer('') == 'unknown'
