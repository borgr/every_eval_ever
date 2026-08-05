import re
import tomllib
from pathlib import Path

import pytest

from every_eval_ever.helpers import schema as helper_schema
from every_eval_ever.schema import get_schema_version

PYPROJECT = Path(__file__).resolve().parents[1] / 'pyproject.toml'
BUNDLE_EXTRA = 'all'
# `every-eval-ever[inspect]`, capturing the extra names inside the brackets.
_SELF_REFERENCE = re.compile(r'^\s*([A-Za-z0-9._-]+)\s*\[([^\]]*)\]')


def test_helper_schema_version_is_independent_of_checkout_layout(
    monkeypatch, tmp_path: Path
) -> None:
    installed_module = (
        tmp_path
        / 'site-packages'
        / 'every_eval_ever'
        / 'helpers'
        / 'schema.py'
    )
    monkeypatch.setattr(helper_schema, '__file__', str(installed_module))

    assert helper_schema._load_schema_version() == get_schema_version()


def _normalize(name: str) -> str:
    """Normalize a package or extra name the way pip and PEP 685 do."""
    return re.sub(r'[-_.]+', '-', name).strip().lower()


def _bundled_extras(
    project_name: str, extras: dict[str, list[str]], bundle: str
) -> set[str]:
    """Collect the extras a bundle extra pulls in, following self-references."""
    seen: set[str] = set()
    pending = [bundle]
    while pending:
        current = pending.pop()
        for requirement in extras.get(current, []):
            match = _SELF_REFERENCE.match(requirement)
            if match is None or _normalize(match.group(1)) != project_name:
                continue  # A third-party requirement, not a self-reference.
            for referenced in match.group(2).split(','):
                referenced = _normalize(referenced)
                if referenced and referenced not in seen:
                    seen.add(referenced)
                    pending.append(referenced)
    return seen


def test_all_extra_installs_every_optional_extra() -> None:
    """`pip install every-eval-ever[all]` must really mean all of them.

    CI installs with ``uv sync --all-extras``, which resolves every extra
    directly and therefore passes even when ``all`` has fallen behind. The
    only people who see the gap are users who installed the published
    ``[all]`` bundle and then hit an ImportError on an adapter that CI
    exercises happily.
    """
    if not PYPROJECT.is_file():
        pytest.skip(f'pyproject.toml not available: {PYPROJECT}')

    pyproject = tomllib.loads(PYPROJECT.read_text(encoding='utf-8'))
    project = pyproject['project']
    project_name = _normalize(project['name'])
    extras = {
        _normalize(name): requirements
        for name, requirements in project.get(
            'optional-dependencies', {}
        ).items()
    }

    assert BUNDLE_EXTRA in extras, (
        f'pyproject.toml declares no {BUNDLE_EXTRA!r} extra; either add it or '
        'drop this test along with the promise it checks'
    )
    expected = set(extras) - {BUNDLE_EXTRA}
    missing = expected - _bundled_extras(project_name, extras, BUNDLE_EXTRA)

    assert not missing, (
        f'the {BUNDLE_EXTRA!r} extra does not install: '
        f'{", ".join(sorted(missing))}. Add '
        + ', '.join(f'"{project["name"]}[{name}]"' for name in sorted(missing))
        + f' to [project.optional-dependencies] {BUNDLE_EXTRA} in '
        'pyproject.toml, then re-run `uv lock` so uv.lock records the new '
        'edge.'
    )
