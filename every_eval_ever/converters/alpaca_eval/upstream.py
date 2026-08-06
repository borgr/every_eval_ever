"""Pinned, replayable access to the upstream ``tatsu-lab/alpaca_eval`` repo.

The AlpacaEval leaderboards are CSV tables in the upstream GitHub repository.
Those tables carry the *scores* only — the judge, the judging prompt, the
baseline, the harness version and every model's real identity and generation
settings live in sibling files of the same repo. This module fetches all of
those artefacts **at one pinned git ref** so a conversion is reproducible, and
serialises them into a single JSON payload so a run can be replayed offline
(``--save-raw-json`` / ``--input-json``).
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence
from urllib.parse import quote

import requests
import yaml

from every_eval_ever.helpers.fetch import FetchError, fetch_text

# ---------------------------------------------------------------------------
# Upstream repository coordinates
# ---------------------------------------------------------------------------

UPSTREAM_REPO = 'tatsu-lab/alpaca_eval'
UPSTREAM_URL = f'https://github.com/{UPSTREAM_REPO}'

#: Upstream ``main`` at the time of writing. Both leaderboard CSVs have been
#: frozen for longer than that (v1 since 2024-05-12, v2 since 2024-12-27), so
#: pinning costs nothing and makes ``evaluation_id`` stable across reruns.
DEFAULT_UPSTREAM_REF = 'cd543a149df89434d8a54582c0151c0b945c3d20'

_RAW_URL = 'https://raw.githubusercontent.com/{repo}/{ref}/{path}'
_BLOB_URL = 'https://github.com/{repo}/blob/{ref}/{path}'

_VERSION_PATH = 'src/alpaca_eval/__init__.py'
_EVALUATORS_DIR = 'src/alpaca_eval/evaluators_configs'
_MODELS_DIR = 'src/alpaca_eval/models_configs'

_VERSION_RE = re.compile(r'^__version__\s*=\s*[\'"]([^\'"]+)[\'"]', re.M)


def raw_url(path: str, ref: str = DEFAULT_UPSTREAM_REF) -> str:
    """Return the raw.githubusercontent URL for *path* at *ref*."""
    return _RAW_URL.format(repo=UPSTREAM_REPO, ref=ref, path=path)


def blob_url(path: str, ref: str = DEFAULT_UPSTREAM_REF) -> str:
    """Return the human-browsable GitHub URL for *path* at *ref*."""
    return _BLOB_URL.format(repo=UPSTREAM_REPO, ref=ref, path=path)


def model_config_path(slug: str) -> str:
    """Return the repo path of a leaderboard entry's model config."""
    return f'{_MODELS_DIR}/{quote(slug)}/configs.yaml'


def annotator_config_path(annotator: str) -> str:
    """Return the repo path of an annotator (judge) config."""
    return f'{_EVALUATORS_DIR}/{annotator}/configs.yaml'


def judge_prompt_path(prompt_template: str) -> str:
    """Return the repo path of a judge prompt template.

    ``prompt_template`` values inside annotator configs are relative to the
    evaluators-configs directory (e.g. ``alpaca_eval_gpt4/alpaca_eval.txt``).
    """
    return f'{_EVALUATORS_DIR}/{prompt_template}'


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------


def parse_single_entry_yaml(text: str, expected_key: str) -> Dict[str, Any]:
    """Parse an upstream config file that wraps one entry in its own name.

    Both ``models_configs/<slug>/configs.yaml`` and
    ``evaluators_configs/<annotator>/configs.yaml`` hold a single top-level key
    (usually, but not always, equal to the directory name) mapping to the
    actual config body.

    Raises:
        ValueError: If the document is not a single-entry mapping of mappings.
    """
    payload = yaml.safe_load(text)
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f'expected a non-empty mapping, got {type(payload)}')
    body = payload.get(expected_key)
    if body is None and len(payload) == 1:
        body = next(iter(payload.values()))
    if not isinstance(body, dict):
        raise ValueError(
            f'expected mapping under {expected_key!r}, got {type(body)}'
        )
    return body


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


@dataclass
class LeaderboardSnapshot:
    """One leaderboard CSV plus the judge it was produced with."""

    rows: List[Dict[str, str]]
    annotator_config: Dict[str, Any]
    judge_prompt: str
    judge_prompt_path: str


@dataclass
class UpstreamSnapshot:
    """Everything a conversion needs from upstream, at one pinned ref."""

    ref: str = DEFAULT_UPSTREAM_REF
    package_version: str = 'unknown'
    leaderboards: Dict[str, LeaderboardSnapshot] = field(default_factory=dict)
    #: slug -> ``models_configs/<slug>/configs.yaml`` body
    model_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    #: leaderboard slugs with no upstream model config (or an unparseable one)
    missing_model_configs: Dict[str, str] = field(default_factory=dict)

    # -- serialisation ------------------------------------------------------

    def to_payload(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict of the whole snapshot."""
        return {
            'ref': self.ref,
            'package_version': self.package_version,
            'leaderboards': {
                version: {
                    'rows': board.rows,
                    'annotator_config': board.annotator_config,
                    'judge_prompt': board.judge_prompt,
                    'judge_prompt_path': board.judge_prompt_path,
                }
                for version, board in self.leaderboards.items()
            },
            'model_configs': self.model_configs,
            'missing_model_configs': self.missing_model_configs,
        }

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> 'UpstreamSnapshot':
        """Rebuild a snapshot from :meth:`to_payload` output."""
        boards = {}
        for version, board in (payload.get('leaderboards') or {}).items():
            boards[version] = LeaderboardSnapshot(
                rows=list(board.get('rows') or []),
                annotator_config=dict(board.get('annotator_config') or {}),
                judge_prompt=board.get('judge_prompt') or '',
                judge_prompt_path=board.get('judge_prompt_path') or '',
            )
        return cls(
            ref=payload.get('ref') or DEFAULT_UPSTREAM_REF,
            package_version=payload.get('package_version') or 'unknown',
            leaderboards=boards,
            model_configs=dict(payload.get('model_configs') or {}),
            missing_model_configs=dict(
                payload.get('missing_model_configs') or {}
            ),
        )


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


def _parse_csv(text: str) -> List[Dict[str, str]]:
    import csv
    import io

    return list(csv.DictReader(io.StringIO(text)))


def populate_snapshot(
    snapshot: UpstreamSnapshot,
    boards: Dict[str, Dict[str, Any]],
    slugs_of: Any,
    max_workers: int = 8,
) -> UpstreamSnapshot:
    """Fetch the upstream artefacts *boards* needs, into *snapshot*.

    Anything already present in *snapshot* is left alone, so converting v1 and
    then v2 fetches each leaderboard once and each model config once.

    Args:
        snapshot: Snapshot to fill; its ``ref`` is the ref that gets fetched.
        boards: ``{version: leaderboard config}`` (see ``adapter.LEADERBOARDS``);
            each config needs ``csv_path`` and ``annotator``.
        slugs_of: Callable mapping the parsed CSV rows of one board to the
            leaderboard slugs whose model configs should be fetched.
        max_workers: Parallel connections used for the per-model configs.

    Returns:
        The same *snapshot*, populated.

    Raises:
        FetchError: If a leaderboard CSV, annotator config, judge prompt or the
            version file cannot be fetched. Individual *model* configs are
            allowed to be missing and are recorded in ``missing_model_configs``.
    """
    ref = snapshot.ref
    with requests.Session() as session:
        if snapshot.package_version in ('', 'unknown', None):
            version_text = fetch_text(
                raw_url(_VERSION_PATH, ref), session=session
            )
            match = _VERSION_RE.search(version_text)
            snapshot.package_version = match.group(1) if match else 'unknown'

        slugs: List[str] = []
        for version, cfg in boards.items():
            board = snapshot.leaderboards.get(version)
            if board is None:
                rows = _parse_csv(
                    fetch_text(raw_url(cfg['csv_path'], ref), session=session)
                )
                annotator_path = annotator_config_path(cfg['annotator'])
                annotator_config = parse_single_entry_yaml(
                    fetch_text(raw_url(annotator_path, ref), session=session),
                    cfg['annotator'],
                )
                prompt_template = annotator_config.get('prompt_template')
                if not isinstance(prompt_template, str) or not prompt_template:
                    raise FetchError(
                        f'annotator {cfg["annotator"]!r} declares no '
                        f'prompt_template at {annotator_path}'
                    )
                prompt_path = judge_prompt_path(prompt_template)
                board = LeaderboardSnapshot(
                    rows=rows,
                    annotator_config=annotator_config,
                    judge_prompt=fetch_text(
                        raw_url(prompt_path, ref), session=session
                    ),
                    judge_prompt_path=prompt_path,
                )
                snapshot.leaderboards[version] = board
            slugs.extend(slugs_of(board.rows))

        pending = [
            slug
            for slug in _unique(slugs)
            if slug not in snapshot.model_configs
            and slug not in snapshot.missing_model_configs
        ]
        if pending:
            _fetch_model_configs(
                snapshot, pending, ref, session, max_workers
            )
    return snapshot


def fetch_snapshot(
    boards: Dict[str, Dict[str, Any]],
    slugs_of: Any,
    ref: str = DEFAULT_UPSTREAM_REF,
    max_workers: int = 8,
) -> UpstreamSnapshot:
    """Fetch every upstream artefact needed to convert *boards*, at *ref*."""
    return populate_snapshot(
        UpstreamSnapshot(ref=ref), boards, slugs_of, max_workers
    )


def _unique(values: Iterable[str]) -> List[str]:
    seen: Dict[str, None] = {}
    for value in values:
        if value and value not in seen:
            seen[value] = None
    return list(seen)


def _fetch_model_configs(
    snapshot: UpstreamSnapshot,
    slugs: Sequence[str],
    ref: str,
    session: requests.Session,
    max_workers: int,
) -> None:
    def _one(slug: str) -> tuple:
        path = model_config_path(slug)
        try:
            text = fetch_text(raw_url(path, ref), session=session)
        except FetchError as exc:
            return slug, None, f'not fetchable: {exc}'
        try:
            return slug, parse_single_entry_yaml(text, slug), None
        except (yaml.YAMLError, ValueError) as exc:
            return slug, None, f'unparseable config: {exc}'

    workers = max(1, min(max_workers, len(slugs) or 1))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for slug, config, error in pool.map(_one, slugs):
            if config is None:
                snapshot.missing_model_configs[slug] = error or 'missing'
            else:
                snapshot.model_configs[slug] = config


def model_config_reference(
    slug: str, ref: str = DEFAULT_UPSTREAM_REF
) -> Optional[str]:
    """Return the browsable URL of a leaderboard entry's model config."""
    return blob_url(model_config_path(slug), ref)
