"""Unified developer/organization extraction from model names."""

import re
from typing import Optional

DEVELOPER_PATTERNS = {
    # OpenAI models
    'gpt': 'openai',
    'text-davinci': 'openai',
    'text-curie': 'openai',
    'text-babbage': 'openai',
    'text-ada': 'openai',
    'davinci': 'openai',
    'curie': 'openai',
    'babbage': 'openai',
    'ada': 'openai',
    'o1': 'openai',
    'o3': 'openai',
    'o4': 'openai',
    # Anthropic models
    'claude': 'anthropic',
    # Google models
    'gemini': 'google',
    'gemma': 'google',
    'palm': 'google',
    't5': 'google',
    'ul2': 'google',
    'text-bison': 'google',
    'text-unicorn': 'google',
    # Meta models
    'llama': 'meta-llama',
    'opt': 'facebook',  # OPT ships under facebook/, not meta-llama/
    # Mistral models
    'mistral': 'mistralai',
    'mixtral': 'mistralai',
    'devstral': 'mistralai',
    # Alibaba models
    'qwen': 'Qwen',
    # Microsoft models
    'phi': 'microsoft',
    'tnlg': 'microsoft',
    # AI21 models
    'j1': 'ai21labs',
    'j2': 'ai21labs',
    'jamba': 'ai21labs',
    'jurassic': 'ai21labs',
    # Cohere models
    'command': 'CohereForAI',
    'cohere': 'CohereForAI',
    'aya': 'CohereForAI',
    'granite': 'ibm',
    # Other providers
    'falcon': 'tiiuae',
    'bloom': 'bigscience',
    't0pp': 'bigscience',
    'pythia': 'EleutherAI',
    'gpt-j': 'EleutherAI',
    'gpt-neox': 'EleutherAI',
    'luminous': 'Aleph-Alpha',
    'mpt': 'mosaicml',
    'redpajama': 'togethercomputer',
    'vicuna': 'lmsys',
    'alpaca': 'tatsu-lab',
    'palmyra': 'Writer',
    'instructpalmyra': 'Writer',
    'yalm': 'yandex',
    'glm': 'zai-org',
    'deepseek': 'deepseek-ai',
    'yi': '01-ai',
    'solar': 'upstage',
    'arctic': 'Snowflake',
    'dbrx': 'databricks',
    'olmo': 'allenai',
    'nova': 'amazon',
    'grok': 'xai',
    'kimi': 'moonshotai',
}

#: Families matched only where the name *starts* with the pattern.
#:
#: ``DEVELOPER_PATTERNS`` also matches a pattern that follows a hyphen, which
#: is what lets ``Flan-T5`` answer ``google``. For the families below that
#: infix match would claim other groups' work: ``KnowRL-Nemotron-1.5B`` is not
#: NVIDIA's, ``BIGBIRD-RoBERTa`` is not Meta's, and ``B3D-RWKV-7.2B`` is not
#: the RWKV project's. Naming the family at the head of the name is the part
#: that identifies a release, so only that is honoured here.
#:
#: Every entry was checked against the model names Papers with Code publishes
#: for it -- a family whose head match also claims a third party's system
#: (``LLaDA`` ships under two orgs; ``MiniCPM-SALA`` is not OpenBMB's;
#: ``Chameleon-SFT`` and ``xLSTM-SENet2`` are third parties' variants) is
#: deliberately absent, because a wrong developer is worse than a dropped row.
HEAD_ONLY_PATTERNS = {
    # Baidu
    'ernie': 'baidu',
    # Meituan
    'longcat': 'meituan-longcat',
    # Meta
    'roberta': 'facebook',
    'bart': 'facebook',
    'flava': 'facebook',
    # Google
    'albert': 'google',
    'electra': 'google',
    'pegasus': 'google',
    'paligemma': 'google',
    # Microsoft
    'deberta': 'microsoft',
    'mpnet': 'microsoft',
    'bitnet': 'microsoft',
    # NVIDIA
    'nemotron': 'nvidia',
    'megatron': 'nvidia',
    # Mistral
    'ministral': 'mistralai',
    'pixtral': 'mistralai',
    # Others
    'distilbert': 'distilbert',
    'xlm-r': 'facebook',
    'apertus': 'swiss-ai',
    'dots.llm': 'rednote-hilab',
    'powerlm': 'ibm-granite',
    'helium': 'kyutai',
    'funnel-transformer': 'funnel-transformer',
    'aguvis': 'xlangai',
    'exaone': 'LGAI-EXAONE',
    'rwkv': 'RWKV',
    'idefics': 'HuggingFaceM4',
    'ovis': 'AIDC-AI',
    'hunyuan': 'tencent',
}


#: Families that must not run on into another letter to count as a match.
#:
#: Several patterns are also ordinary word fragments, so a plain prefix test
#: reads a third party's system as the family's: ``AdaFace`` and
#: ``AdapterTune`` are not OpenAI's ``ada``, ``OptiMer`` is not Meta's
#: ``OPT``, ``LlamaGen`` is not Meta's, ``QwenLong`` ships under
#: ``Tongyi-Zhiwen``, ``MPTSNet`` is not MosaicML's and ``YingLong`` is not
#: 01-ai's ``Yi``. Requiring the character after the family to be anything but
#: a letter keeps the real spellings (``ada-002``, ``opt-1.3b``, ``Qwen3``,
#: ``Phi-4``, ``Yi-34B``) and refuses those.
#:
#: The families left out of this set are ones whose own releases do continue in
#: letters -- ``T5Gemma``, ``OLMoE``, ``olmOCR``, ``KimiVL``, ``HunyuanOCR`` --
#: where the boundary would drop a correctly attributed row.
BOUNDED_PATTERNS = frozenset({
    'ada', 'babbage', 'curie', 'davinci', 'gpt', 'o1', 'o3', 'o4',
    'llama', 'opt', 'mistral', 'qwen', 'phi', 'yi', 'glm', 'mpt',
    'nova', 'solar', 'arctic', 'command', 'aya', 'j1', 'j2',
})


def _family_matcher(pattern: str, head_only: bool) -> 're.Pattern[str]':
    """One family's matcher, built once per pattern.

    ``(?:^|-)`` is the infix rule ``DEVELOPER_PATTERNS`` has always had: a
    family at the head of the name, or following a hyphen, so ``Flan-T5``
    answers ``google``.
    """
    head = r'^' if head_only else r'(?:^|-)'
    tail = r'(?![a-z])' if pattern in BOUNDED_PATTERNS else ''
    return re.compile(f'{head}{re.escape(pattern)}{tail}')


_FAMILY_MATCHERS = tuple(
    (_family_matcher(pattern, head_only=False), developer)
    for pattern, developer in DEVELOPER_PATTERNS.items()
) + tuple(
    (_family_matcher(pattern, head_only=True), developer)
    for pattern, developer in HEAD_ONLY_PATTERNS.items()
)


#: A name that is already ``namespace/model``, so its prefix *is* the developer.
#:
#: A slash on its own does not say that. Leaderboards spell a patch size
#: ``ViT-L/16``, a variant ``SR-DiT-B/1``, and a setting ``Claude Opus 5
#: (w/ tools)``, and splitting those on the slash invents the publishers
#: ``ViT-L``, ``SR-DiT-B`` and ``Claude Opus 5 (w``. What separates an id from
#: all three is that it carries no spaces or brackets and never names bare
#: digits after the namespace. Anything further in is left alone, because a
#: submission may address a model as ``org/family/model:revision``.
HF_REPO_ID_RE = re.compile(
    r'^[A-Za-z0-9][A-Za-z0-9._-]*/(?!\d+(?:$|/))[^\s()\[\]]+$'
)


def get_developer(model_name: str) -> str:
    """
    Extract developer/organization name from a model name.

    Uses a two-step approach:
    1. If model_name contains '/', use the prefix as the developer
    2. Otherwise, pattern match against known model families

    Args:
        model_name: The model name (e.g., "meta-llama/Llama-3-8B" or "gpt-4")

    Both steps answer with the **publishing namespace**, never the parent
    company, because that is what the datastore's developer folder is (see
    ``datastore_path_components``). So the two agree for one model however a
    source spells it: ``Qwen/Qwen3-32B`` and ``qwen3-32b`` both give ``Qwen``.
    ``test_developer.py`` pins that agreement pair by pair — a new pattern whose
    value is a company rather than a namespace fails there.

    A bare name resolved through the eval-card-registry is still better where an
    adapter can afford the lookup, because this table only knows the families
    listed in it.

    Returns:
        Developer name (lowercase), or "unknown" if not recognized

    Examples:
        >>> get_developer("meta-llama/Llama-3-8B")
        "meta-llama"
        >>> get_developer("gpt-4-turbo")
        "openai"
        >>> get_developer("claude-3-opus")
        "anthropic"
        >>> get_developer("some-random-model")
        "unknown"
    """
    if not model_name:
        return 'unknown'

    # If already has org prefix (e.g., "meta-llama/Llama-3-8B"), use it
    if HF_REPO_ID_RE.match(model_name):
        return model_name.split('/')[0]

    # Pattern match against known model families
    lower_name = model_name.lower()
    for matcher, developer in _FAMILY_MATCHERS:
        if matcher.search(lower_name):
            return developer

    return 'unknown'


def get_model_id(model_name: str, developer: Optional[str] = None) -> str:
    """
    Generate a standardized model ID in the format 'developer/model'.

    Args:
        model_name: The model name
        developer: Optional developer override; if not provided, will be extracted

    Returns:
        Model ID in 'developer/model' format

    Examples:
        >>> get_model_id("Llama-3-8B", "meta")
        "meta/Llama-3-8B"
        >>> get_model_id("openai/gpt-4")
        "openai/gpt-4"
    """
    if '/' in model_name:
        return model_name

    dev = developer or get_developer(model_name)
    return f'{dev}/{model_name}'
