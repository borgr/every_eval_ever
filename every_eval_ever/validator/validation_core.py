"""Validation checks shared by the local command and validator bot."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Container
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import ValidationError

from every_eval_ever.eval_types import EvaluationLog
from every_eval_ever.helpers.org_registry import second_name_of
from every_eval_ever.instance_level_types import InstanceLevelEvaluationLog
from every_eval_ever.schema import (
    get_schema_fingerprint as get_schema_fingerprint,
)
from every_eval_ever.schema import get_schema_version as get_schema_version
from every_eval_ever.validator.json_utils import (
    StrictJSONError,
    strict_json_loads,
)

DEFAULT_MAX_ERRORS = 50

_EXPECTED_PATH_PARTS = 5  # data / benchmark / developer / model / filename
_UUID_RE = (
    r'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}'
)
_AGGREGATE_FILE_RE = re.compile(rf'{_UUID_RE}\.json$')
_INSTANCE_FILE_RE = re.compile(rf'{_UUID_RE}_samples\.jsonl$')

_DEPLOYMENT_TYPES = ('self_deployed', 'externally_managed', 'unknown')
_MODEL_AVAILABILITY_TYPES = ('open_weights', 'closed_weights', 'unknown')
_INVALID_PATH_COMPONENT_CHARS = re.compile(r'[<>:"\\|?*\x00-\x1f]')
_WINDOWS_RESERVED_NAMES = {
    'CON',
    'PRN',
    'AUX',
    'NUL',
    *(f'COM{index}' for index in range(1, 10)),
    *(f'LPT{index}' for index in range(1, 10)),
}


@dataclass
class ValidationReport:
    """Result of validating a single file."""

    file_path: Path
    valid: bool
    errors: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[dict[str, Any]] = field(default_factory=list)
    file_type: str = ''
    line_count: int = 0


@dataclass(frozen=True)
class ValidationContext:
    """Repository information needed by path and companion checks."""

    repo_path: str
    available_files: Container[str] = field(default_factory=frozenset)
    read_repo_file: Callable[[str], str] | None = None


@dataclass(frozen=True)
class InstanceFileSummary:
    """Cross-file values collected while validating a JSONL file."""

    line_count: int
    evaluation_ids: frozenset[str]
    model_ids: frozenset[str]
    content_valid: bool = True


CheckScope = Literal['aggregate', 'instance', 'file']
CheckSeverity = Literal['error', 'warning']
ValidationPayload = dict[str, Any] | InstanceFileSummary | None


@dataclass(frozen=True)
class ValidationCheck:
    """A named validation check registered with the shared runner."""

    name: str
    scope: CheckScope
    severity: CheckSeverity
    run: Callable[[ValidationContext, ValidationPayload], list[str]]


class SemanticCheckError(RuntimeError):
    """Raised when a registered semantic check cannot complete."""


@dataclass
class SemanticCheckReport:
    """Blocking and advisory findings produced by registered checks."""

    errors: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[dict[str, Any]] = field(default_factory=list)


def _format_loc(loc: tuple[Any, ...]) -> str:
    parts = []
    for part in loc:
        if isinstance(part, int):
            parts.append(f'[{part}]')
        else:
            if parts:
                parts.append(f' -> {part}')
            else:
                parts.append(str(part))
    return ''.join(parts) if parts else '(root)'


def pydantic_errors_to_dicts(exc: ValidationError) -> list[dict[str, Any]]:
    """Convert Pydantic errors to the report format used by the CLI and Space."""
    errors: list[dict[str, Any]] = []
    for err in exc.errors():
        errors.append(
            {
                'loc': _format_loc(err['loc']),
                'msg': err['msg'],
                'type': err['type'],
                'input': err.get('input'),
            }
        )
    return errors


def warning_to_dict(message: str) -> dict[str, str]:
    """Convert a grouped warning string into a structured report warning."""
    if ': ' in message:
        loc, msg = message.split(': ', 1)
        return {'loc': loc, 'msg': msg, 'type': 'semantic_warning'}
    return {'loc': '', 'msg': message, 'type': 'semantic_warning'}


def semantic_error_to_dict(message: str) -> dict[str, str]:
    """Convert a grouped semantic-rule message into a blocking error."""
    if ': ' in message:
        loc, msg = message.split(': ', 1)
        return {'loc': loc, 'msg': msg, 'type': 'semantic_rule_error'}
    return {'loc': '', 'msg': message, 'type': 'semantic_rule_error'}


def format_warning(warning: dict[str, Any]) -> str:
    """Format a warning dict as the signature used for grouping."""
    loc = warning.get('loc')
    msg = warning.get('msg', '')
    return f'{loc}: {msg}' if loc else str(msg)


def format_error(error: dict[str, Any]) -> str:
    loc = error.get('loc')
    msg = error.get('msg', '')
    return f'{loc}: {msg}' if loc else str(msg)


def _json_error_details(
    exc: json.JSONDecodeError | StrictJSONError,
    *,
    line_num: int | None = None,
) -> tuple[str, str]:
    if isinstance(exc, json.JSONDecodeError):
        source_line = line_num if line_num is not None else exc.lineno
        return f'line {source_line}, col {exc.colno}', exc.msg
    location = f'line {line_num}' if line_num is not None else '(json)'
    return location, str(exc)


def check_path_structure(repo_path: str) -> list[str]:
    """Enforce aggregate and instance datastore paths."""
    parts = repo_path.split('/')

    if len(parts) != _EXPECTED_PATH_PARTS:
        return [
            'Unexpected path depth: expected '
            "'data/benchmark/developer/model/uuid.json' or "
            "'data/benchmark/developer/model/uuid_samples.jsonl', "
            f"got {len(parts)} components in '{repo_path}'"
        ]

    if (
        repo_path.startswith('/')
        or '\\' in repo_path
        or any(part in {'', '.', '..'} for part in parts)
    ):
        return [
            'Path must be a clean repository-relative path without empty, '
            f"current, or parent components: '{repo_path}'"
        ]

    if parts[0] != 'data':
        return [f"Path does not start with 'data/': '{repo_path}'"]

    reserved_components = [
        component for component in parts[1:4] if component == 'data'
    ]
    if reserved_components:
        return [
            'Collection, developer, and model path components cannot use '
            f"the reserved datastore name 'data': '{repo_path}'"
        ]

    for component in parts[1:4]:
        if (
            _INVALID_PATH_COMPONENT_CHARS.search(component)
            or component.endswith(('.', ' '))
            or component.split('.', 1)[0].upper() in _WINDOWS_RESERVED_NAMES
        ):
            return [
                'Collection, developer, and model path components must be '
                'portable filesystem names; got '
                f'{component!r} in {repo_path!r}'
            ]

    filename = parts[4]
    if not (
        _AGGREGATE_FILE_RE.fullmatch(filename)
        or _INSTANCE_FILE_RE.fullmatch(filename)
    ):
        return [
            f"Filename '{filename}' does not match '{{UUID4}}.json' or "
            f"'{{UUID4}}_samples.jsonl' in '{repo_path}'"
        ]

    return []


def resolve_companion_repo_path(
    repo_path: str, aggregate_data: dict[str, Any]
) -> str | None:
    """Return the one companion path allowed for an aggregate."""
    detail = aggregate_data.get('detailed_evaluation_results')
    if detail is None:
        return None
    if not isinstance(detail, dict):
        raise ValueError('detailed_evaluation_results must be an object')

    reference = detail.get('file_path')
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError(
            'detailed_evaluation_results.file_path: missing or blank companion path'
        )

    aggregate_path = PurePosixPath(repo_path)
    expected_path = (
        aggregate_path.parent / f'{aggregate_path.stem}_samples.jsonl'
    ).as_posix()
    normalized_reference = reference.strip()
    if normalized_reference != expected_path:
        raise ValueError(
            'detailed_evaluation_results.file_path: expected exactly '
            f'{expected_path!r} so the aggregate and samples share one UUID '
            f'and folder, got {reference!r}'
        )

    return expected_path


def _aggregate_repo_path_for_samples(repo_path: str) -> str | None:
    sample_path = PurePosixPath(repo_path)
    suffix = '_samples.jsonl'
    if not sample_path.name.endswith(suffix):
        return None
    aggregate_name = f'{sample_path.name[: -len(suffix)]}.json'
    return (sample_path.parent / aggregate_name).as_posix()


def _summary_identifier(value: Any) -> str:
    return value if isinstance(value, str) else repr(value)


def _summarize_jsonl_text(content: str) -> InstanceFileSummary:
    evaluation_ids: set[str] = set()
    model_ids: set[str] = set()
    line_count = 0
    for source_line, line in enumerate(content.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        line_count += 1
        try:
            row = strict_json_loads(stripped)
        except (json.JSONDecodeError, StrictJSONError) as exc:
            _, message = _json_error_details(exc, line_num=source_line)
            raise ValueError(
                f'samples line {source_line} is invalid JSON: {message}'
            ) from exc
        if not isinstance(row, dict):
            raise ValueError(
                f'samples line {source_line} must contain a JSON object'
            )
        evaluation_ids.add(_summary_identifier(row.get('evaluation_id')))
        model_ids.add(_summary_identifier(row.get('model_id')))
    return InstanceFileSummary(
        line_count=line_count,
        evaluation_ids=frozenset(evaluation_ids),
        model_ids=frozenset(model_ids),
    )


def _compare_aggregate_and_samples(
    aggregate_data: dict[str, Any],
    samples: InstanceFileSummary,
) -> list[str]:
    if not samples.content_valid:
        return []

    errors: list[str] = []
    if samples.line_count == 0:
        errors.append('samples companion must contain at least one JSONL row')
    expected_evaluation_id = aggregate_data.get('evaluation_id')
    unexpected_evaluation_ids = samples.evaluation_ids - {
        _summary_identifier(expected_evaluation_id)
    }
    if unexpected_evaluation_ids:
        errors.append(
            'samples evaluation_id values must match the aggregate '
            f'evaluation_id {expected_evaluation_id!r}; got '
            f'{sorted(unexpected_evaluation_ids)!r}'
        )

    model_info = aggregate_data.get('model_info')
    expected_model_id = (
        model_info.get('id') if isinstance(model_info, dict) else None
    )
    unexpected_model_ids = samples.model_ids - {
        _summary_identifier(expected_model_id)
    }
    if unexpected_model_ids:
        errors.append(
            'samples model_id values must match the aggregate model_info.id '
            f'{expected_model_id!r}; got {sorted(unexpected_model_ids)!r}'
        )

    detail = aggregate_data.get('detailed_evaluation_results')
    total_rows = detail.get('total_rows') if isinstance(detail, dict) else None
    if isinstance(total_rows, int) and total_rows != samples.line_count:
        errors.append(
            'detailed_evaluation_results.total_rows does not match the '
            f'companion file: declared {total_rows}, found {samples.line_count}'
        )
    return errors


def check_companion_exists(
    repo_path: str,
    aggregate_data: dict[str, Any],
    available_files: Container[str],
    read_repo_file: Callable[[str], str] | None = None,
) -> list[str]:
    """Enforce an aggregate's forward and reverse samples relationship."""
    expected_path = (
        PurePosixPath(repo_path).parent
        / f'{PurePosixPath(repo_path).stem}_samples.jsonl'
    ).as_posix()
    detail = aggregate_data.get('detailed_evaluation_results')
    if detail is None:
        if expected_path in available_files:
            return [
                'detailed_evaluation_results is required because sibling '
                f'samples file {expected_path!r} exists'
            ]
        return []

    try:
        resolved_text = resolve_companion_repo_path(repo_path, aggregate_data)
    except ValueError as exc:
        return [str(exc)]
    if resolved_text is None:
        return []

    errors: list[str] = []
    for path_error in check_path_structure(resolved_text):
        errors.append(
            'detailed_evaluation_results.file_path: '
            f'declared companion has invalid datastore path: {path_error}'
        )
    declared_format = detail.get('format')
    if declared_format != 'jsonl':
        errors.append(
            'detailed_evaluation_results.format must be exactly '
            f"'jsonl', got {declared_format!r}"
        )

    if resolved_text not in available_files:
        errors.append(
            'detailed_evaluation_results.file_path: referenced companion '
            f'{resolved_text!r} was not found in the dataset or this batch'
        )
        return errors
    if read_repo_file is None:
        errors.append(
            'detailed_evaluation_results.file_path: companion contents could '
            'not be checked because no repository file reader was provided'
        )
        return errors

    try:
        samples = _summarize_jsonl_text(read_repo_file(resolved_text))
    except (OSError, ValueError) as exc:
        errors.append(
            'detailed_evaluation_results.file_path: could not inspect '
            f'companion {resolved_text!r}: {exc}'
        )
        return errors
    errors.extend(_compare_aggregate_and_samples(aggregate_data, samples))
    return errors


def check_instance_companion(
    repo_path: str,
    samples: InstanceFileSummary,
    available_files: Container[str],
    read_repo_file: Callable[[str], str] | None = None,
) -> list[str]:
    """Require a samples file's aggregate to exist and point back to it."""
    aggregate_path = _aggregate_repo_path_for_samples(repo_path)
    if aggregate_path is None:
        return []
    if aggregate_path not in available_files:
        return [f'samples file requires sibling aggregate {aggregate_path!r}']
    if read_repo_file is None:
        return [
            f'sibling aggregate {aggregate_path!r} could not be checked '
            'because no repository file reader was provided'
        ]

    try:
        aggregate_data = strict_json_loads(read_repo_file(aggregate_path))
    except (OSError, json.JSONDecodeError, StrictJSONError) as exc:
        return [
            f'could not inspect sibling aggregate {aggregate_path!r}: {exc}'
        ]
    if not isinstance(aggregate_data, dict):
        return [
            f'sibling aggregate {aggregate_path!r} must contain a JSON object'
        ]

    detail = aggregate_data.get('detailed_evaluation_results')
    if detail is None:
        return [
            f'sibling aggregate {aggregate_path!r} must declare '
            'detailed_evaluation_results for this samples file'
        ]
    try:
        declared_samples = resolve_companion_repo_path(
            aggregate_path, aggregate_data
        )
    except ValueError as exc:
        return [str(exc)]
    if declared_samples != repo_path:
        return [
            f'sibling aggregate {aggregate_path!r} does not point to '
            f'this samples file {repo_path!r}'
        ]

    errors: list[str] = []
    if isinstance(detail, dict) and detail.get('format') != 'jsonl':
        errors.append(
            'detailed_evaluation_results.format must be exactly '
            f"'jsonl', got {detail.get('format')!r}"
        )
    errors.extend(_compare_aggregate_and_samples(aggregate_data, samples))
    return errors


def check_score_metadata(data: dict[str, Any]) -> list[str]:
    """Validate supplied bounds and require them for continuous metrics."""
    warnings: list[str] = []
    results = data.get('evaluation_results')
    if not isinstance(results, list):
        return warnings

    for index, result in enumerate(results):
        if not isinstance(result, dict):
            continue
        metric = result.get('metric_config')
        if not isinstance(metric, dict):
            continue
        score_type = metric.get('score_type')
        if score_type is not None and (
            not isinstance(score_type, str) or not score_type.strip()
        ):
            warnings.append(
                f"evaluation_results[{index}].metric_config: invalid 'score_type'"
            )

        raw_lo = metric.get('min_score')
        raw_hi = metric.get('max_score')
        lo = _metric_bound(raw_lo)
        hi = _metric_bound(raw_hi)
        requires_bounds = score_type == 'continuous'
        for key, raw_value, value in (
            ('min_score', raw_lo, lo),
            ('max_score', raw_hi, hi),
        ):
            if value is None and (requires_bounds or raw_value is not None):
                warnings.append(
                    f'evaluation_results[{index}].metric_config: missing or '
                    f"invalid '{key}'"
                )

        score_details = result.get('score_details')
        if not isinstance(score_details, dict):
            continue
        score = score_details.get('score')
        if not _is_finite_number(score):
            warnings.append(
                f'evaluation_results[{index}].score_details.score: expected a '
                f'finite number, got {score!r}'
            )
            continue
        uncertainty = score_details.get('uncertainty')
        if isinstance(uncertainty, dict):
            finite_uncertainty_fields = {
                'standard_deviation': uncertainty.get('standard_deviation'),
            }
            standard_error = uncertainty.get('standard_error')
            if isinstance(standard_error, dict):
                finite_uncertainty_fields['standard_error.value'] = (
                    standard_error.get('value')
                )
            confidence_interval = uncertainty.get('confidence_interval')
            if isinstance(confidence_interval, dict):
                for name in ('lower', 'upper', 'confidence_level'):
                    if name in confidence_interval:
                        finite_uncertainty_fields[
                            f'confidence_interval.{name}'
                        ] = confidence_interval.get(name)
            for field_name, value in finite_uncertainty_fields.items():
                if value is not None and not _is_finite_number(value):
                    warnings.append(
                        f'evaluation_results[{index}].score_details.'
                        f'uncertainty.{field_name}: expected a finite number, '
                        f'got {value!r}'
                    )
        if lo is not None and hi is not None and lo > hi:
            warnings.append(
                f'evaluation_results[{index}].metric_config: min_score '
                f'{raw_lo} is greater than max_score {raw_hi}'
            )
            continue
        if lo is not None and hi is not None and (score < lo or score > hi):
            warnings.append(
                f'evaluation_results[{index}]: score {score} is outside '
                f'[min_score={raw_lo}, max_score={raw_hi}]'
            )
    return warnings


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _metric_bound(value: Any) -> float | None:
    """Return a comparable metric bound, including the strict-JSON infinity form."""
    if value == 'Infinity':
        return math.inf
    if value == '-Infinity':
        return -math.inf
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and not math.isnan(value)
    ):
        return float(value)
    return None


def _developer_prefix(model_id: Any) -> str | None:
    """Return the namespace segment of a slash-bearing model identity."""
    if not isinstance(model_id, str) or '/' not in model_id:
        return None
    return model_id.split('/', 1)[0]


def check_developer_slug(data: dict[str, Any]) -> list[str]:
    """Warn when a developer slug splits one publisher across two directories.

    The datastore groups records into one directory per developer (see
    ``helpers.io.datastore_path_components``), so two names for one publisher
    become two publishers and neither listing is complete. Adapters disagree
    today, and the published datastore shows the result: ``mistral`` beside
    ``mistralai``, ``moonshot`` beside ``moonshotai``, ``zhipu`` and
    ``zhipu-ai`` and ``THUDM`` beside ``zai``.

    Which names mean the same organization is decided by the
    **eval-card-registry** vocabulary in ``helpers.org_registry``, not by a map
    kept here. Two consequences are worth stating because they are the whole
    reason for using it:

    - The registry separates an organization's canonical id from the
      HuggingFace namespace it publishes under, so ``meta-llama``, ``qwen``,
      ``deepseek-ai`` and ``zai-org`` are accepted identities rather than
      drift. A check written against a local developer map gets this wrong:
      ``get_developer('Llama-3-8B')`` is ``meta``, and comparing with that
      would flag every record correctly filed under ``meta-llama``. The
      registry fills that field in for 11 of its 1166 organizations so far, so a
      namespace it has *not* recorded but does carry as an alias is reported
      like any other second name — ``MiniMaxAI`` for ``minimax``,
      ``CohereLabs`` for ``cohere``. In the published datastore those records
      are filed under the canonical spelling anyway (there is no
      ``MiniMaxAI/``, ``CohereLabs/`` or ``XiaomiMiMo/`` directory in it), so
      the warning still names the directory that exists; the durable fix is
      ``hf_org`` upstream, which serves every consumer.
    - Only *second names* are flagged — a confirmed registry alias that is a
      genuinely different name, including a model family standing in for its
      publisher. Case and punctuation variants (``Anthropic`` for
      ``anthropic``, ``snowflake`` for ``Snowflake``) are left alone: the
      registry aims for HuggingFace-true casing and HuggingFace is not
      internally consistent, so its preferred spelling is not evidence about
      which directory this datastore already uses.

    The message names both spellings and does not pick one. The registry's
    canonical id is an *entity* id, not a directory name — it keeps the
    HuggingFace namespace in a separate field precisely because the two differ
    — and in the published datastore it is often the rarer spelling of the two
    (``zai`` appears in 2 collections against 11 for ``zhipu``; ``mistralai``
    in 27 against 28 for ``mistral``). Naming it as the destination would move
    records toward the minority directory, which is the split this check exists
    to prevent. Choosing one spelling per publisher is a datastore-wide
    decision; a per-file warning can only say that two of them are in play.

    Both ``model_info.id``'s namespace prefix and ``model_info.developer`` are
    checked, and one warning names every field holding the spelling, so a
    rename does not warn again on the next run. Only one of the two decides the
    directory, and only that one is told that it splits it.

    The cost is silence on any organization the registry has not seen — an
    unrecognized slug is assumed to be somebody's real organization, and
    widening coverage means adding an alias to the registry, which serves
    every other consumer too.

    A warning, not an error: these records are already published, the fix is a
    rename that changes join keys, and the name a project treats as primary is
    an editorial call rather than a schema violation.
    """
    model_info = data.get('model_info')
    if not isinstance(model_info, dict):
        return []

    # Both fields carry a publisher name, so both are checked, and every field
    # holding one spelling is named together: renaming only the one that
    # decides the directory would leave the other to warn on the next run.
    prefix = _developer_prefix(model_info.get('id'))
    declared: dict[str, list[str]] = {}
    for location, value in (
        ('model_info.id', prefix),
        ('model_info.developer', model_info.get('developer')),
    ):
        if isinstance(value, str) and value.strip():
            declared.setdefault(value.strip(), []).append(location)

    # Only one of the two decides the directory: the id's namespace prefix when
    # it has one, model_info.developer when the id is flat (see
    # helpers.io.datastore_path_components). The other still names the
    # publisher, so it is still reported — with the consequence it really has.
    directory_field = 'model_info.id' if prefix else 'model_info.developer'
    warnings: list[str] = []
    for slug, locations in declared.items():
        canonical = second_name_of(slug)
        if canonical is None:
            continue
        consequence = (
            'publishing under both puts one developer in two datastore '
            'directories, and neither listing is complete'
            if directory_field in locations
            else 'the id prefix decides the directory here, so this field '
            'does not split it, but anything grouping records by developer '
            'sees two publishers'
        )
        warnings.append(
            f'{" and ".join(locations)}: {slug!r} and {canonical!r} are the '
            'same organization in the eval-card-registry. Use whichever '
            f'spelling this collection already uses — {consequence}'
        )
    return warnings


def check_model_deployment(data: dict[str, Any]) -> list[str]:
    """Require independent deployment-control and weight-availability axes.

    ``deployment_type`` describes who controlled the inference deployment;
    ``model_availability`` describes whether model weights are available.
    Neither value constrains the other. This rule deliberately performs no
    provider-specific existence check.
    """
    warnings: list[str] = []

    def check_one(model_info: Any, location: str) -> None:
        if not isinstance(model_info, dict):
            return
        details = model_info.get('additional_details')
        if not isinstance(details, dict):
            details = {}

        deployment_type = details.get('deployment_type')
        if deployment_type is None:
            warnings.append(
                f"{location}.additional_details: missing 'deployment_type' "
                f'(expected {"|".join(_DEPLOYMENT_TYPES)})'
            )
        elif deployment_type not in _DEPLOYMENT_TYPES:
            warnings.append(
                f'{location}.additional_details.deployment_type: expected '
                f'one of {list(_DEPLOYMENT_TYPES)}, got '
                f'{deployment_type!r}'
            )

        availability = details.get('model_availability')
        if availability is None:
            warnings.append(
                f'{location}.additional_details: missing '
                f"'model_availability' (expected "
                f'{"|".join(_MODEL_AVAILABILITY_TYPES)})'
            )
        elif availability not in _MODEL_AVAILABILITY_TYPES:
            warnings.append(
                f'{location}.additional_details.model_availability: '
                f'expected one of {list(_MODEL_AVAILABILITY_TYPES)}, got '
                f'{availability!r}'
            )

    check_one(data.get('model_info'), 'model_info')
    results = data.get('evaluation_results')
    if isinstance(results, list):
        for result_index, result in enumerate(results):
            if not isinstance(result, dict):
                continue
            metric = result.get('metric_config')
            scoring = (
                metric.get('llm_scoring') if isinstance(metric, dict) else None
            )
            judges = (
                scoring.get('judges') if isinstance(scoring, dict) else None
            )
            if not isinstance(judges, list):
                continue
            for judge_index, judge in enumerate(judges):
                if not isinstance(judge, dict):
                    continue
                check_one(
                    judge.get('model_info'),
                    'evaluation_results'
                    f'[{result_index}].metric_config.llm_scoring.judges'
                    f'[{judge_index}].model_info',
                )
    return warnings


def _file_check_path(
    context: ValidationContext, data: ValidationPayload
) -> list[str]:
    return check_path_structure(context.repo_path)


def _aggregate_check_companion(
    context: ValidationContext, data: ValidationPayload
) -> list[str]:
    if not isinstance(data, dict):
        return []
    return check_companion_exists(
        context.repo_path,
        data,
        context.available_files,
        context.read_repo_file,
    )


def _instance_check_companion(
    context: ValidationContext, data: ValidationPayload
) -> list[str]:
    if not isinstance(data, InstanceFileSummary):
        return []
    return check_instance_companion(
        context.repo_path,
        data,
        context.available_files,
        context.read_repo_file,
    )


def _aggregate_check_score_metadata(
    context: ValidationContext, data: ValidationPayload
) -> list[str]:
    if not isinstance(data, dict):
        return []
    return check_score_metadata(data)


def _aggregate_check_model_deployment(
    context: ValidationContext, data: ValidationPayload
) -> list[str]:
    if not isinstance(data, dict):
        return []
    return check_model_deployment(data)


def _aggregate_check_developer_slug(
    context: ValidationContext, data: ValidationPayload
) -> list[str]:
    if not isinstance(data, dict):
        return []
    return check_developer_slug(data)


REGISTERED_CHECKS: tuple[ValidationCheck, ...] = (
    ValidationCheck('path structure', 'file', 'error', _file_check_path),
    ValidationCheck(
        'companion file', 'aggregate', 'error', _aggregate_check_companion
    ),
    ValidationCheck(
        'aggregate file', 'instance', 'error', _instance_check_companion
    ),
    ValidationCheck(
        'score metadata', 'aggregate', 'error', _aggregate_check_score_metadata
    ),
    ValidationCheck(
        'model deployment',
        'aggregate',
        'error',
        _aggregate_check_model_deployment,
    ),
    ValidationCheck(
        'developer slug',
        'aggregate',
        'warning',
        _aggregate_check_developer_slug,
    ),
)


def run_registered_checks(
    context: ValidationContext,
    *,
    file_type: Literal['aggregate', 'instance'],
    data: ValidationPayload,
    checks: tuple[ValidationCheck, ...] = REGISTERED_CHECKS,
) -> SemanticCheckReport:
    """Run registered checks and preserve their explicit severity."""
    report = SemanticCheckReport()
    for check in checks:
        if check.scope not in {'file', file_type}:
            continue
        try:
            messages = check.run(context, data)
        except Exception as exc:
            raise SemanticCheckError(
                f'{check.name} check did not complete: '
                f'{type(exc).__name__}: {exc or "<no detail>"}'
            ) from exc
        if check.severity == 'error':
            report.errors.extend(
                semantic_error_to_dict(message) for message in messages
            )
        elif check.severity == 'warning':
            report.warnings.extend(
                warning_to_dict(message) for message in messages
            )
        else:
            raise SemanticCheckError(
                f'{check.name} check has unsupported severity {check.severity!r}'
            )
    return report


def validate_aggregate(
    file_path: Path,
    *,
    repo_path: str | None = None,
    available_files: Container[str] | None = None,
    read_repo_file: Callable[[str], str] | None = None,
    run_semantic_checks: bool = False,
) -> ValidationReport:
    """Validate one aggregate file and its repository relationship."""
    report = ValidationReport(
        file_path=file_path, valid=True, file_type='aggregate'
    )
    try:
        raw = file_path.read_text(encoding='utf-8')
    except OSError as exc:
        report.valid = False
        report.errors.append(
            {'loc': '(file)', 'msg': str(exc), 'type': 'io_error'}
        )
        return report

    try:
        loaded = strict_json_loads(raw)
    except (json.JSONDecodeError, StrictJSONError) as exc:
        location, message = _json_error_details(exc)
        report.valid = False
        report.errors.append(
            {
                'loc': location,
                'msg': message,
                'type': 'json_parse_error',
            }
        )
        return report

    data = loaded if isinstance(loaded, dict) else None
    try:
        EvaluationLog.model_validate(loaded)
    except ValidationError as exc:
        report.valid = False
        report.errors = pydantic_errors_to_dicts(exc)

    if run_semantic_checks:
        if repo_path is None:
            report.valid = False
            report.errors.append(
                {
                    'loc': '(semantic checks)',
                    'msg': 'repo_path is required for repository validation',
                    'type': 'semantic_check_error',
                }
            )
            return report
        if available_files is None:
            available_files = frozenset({repo_path})
        context = ValidationContext(
            repo_path=repo_path,
            available_files=available_files,
            read_repo_file=read_repo_file,
        )
        try:
            semantic_report = run_registered_checks(
                context, file_type='aggregate', data=data
            )
            report.errors.extend(semantic_report.errors)
            report.warnings.extend(semantic_report.warnings)
            if semantic_report.errors:
                report.valid = False
        except SemanticCheckError as exc:
            report.valid = False
            report.errors.append(
                {
                    'loc': '(semantic checks)',
                    'msg': str(exc),
                    'type': 'semantic_check_error',
                }
            )

    return report


def _validate_instance_line(
    line: str, line_num: int
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    try:
        data = strict_json_loads(line)
    except (json.JSONDecodeError, StrictJSONError) as exc:
        location, message = _json_error_details(exc, line_num=line_num)
        return (
            [
                {
                    'loc': location,
                    'msg': message,
                    'type': 'json_parse_error',
                }
            ],
            None,
        )

    try:
        InstanceLevelEvaluationLog.model_validate(data)
    except ValidationError as exc:
        errors = pydantic_errors_to_dicts(exc)
        for error in errors:
            error['loc'] = f'line {line_num} -> {error["loc"]}'
        return errors, data if isinstance(data, dict) else None

    return [], data if isinstance(data, dict) else None


def validate_instance_file(
    file_path: Path,
    max_errors: int = DEFAULT_MAX_ERRORS,
    *,
    repo_path: str | None = None,
    available_files: Container[str] | None = None,
    read_repo_file: Callable[[str], str] | None = None,
    run_semantic_checks: bool = False,
) -> ValidationReport:
    """Validate one JSONL file and its repository relationship."""
    report = ValidationReport(
        file_path=file_path, valid=True, file_type='instance'
    )
    try:
        handle = file_path.open(encoding='utf-8')
    except OSError as exc:
        report.valid = False
        report.errors.append(
            {'loc': '(file)', 'msg': str(exc), 'type': 'io_error'}
        )
        return report

    evaluation_ids: set[str] = set()
    model_ids: set[str] = set()
    content_valid = True
    with handle:
        for line_num, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            report.line_count += 1
            line_errors, data = _validate_instance_line(stripped, line_num)
            if data is not None:
                evaluation_ids.add(
                    _summary_identifier(data.get('evaluation_id'))
                )
                model_ids.add(_summary_identifier(data.get('model_id')))
            if not line_errors:
                continue

            report.valid = False
            content_valid = False
            remaining = max_errors - len(report.errors)
            if remaining <= 0:
                report.errors.append(
                    {
                        'loc': '(truncated)',
                        'msg': (
                            f'Error limit reached ({max_errors}). '
                            'Use --max-errors to increase.'
                        ),
                        'type': 'truncated',
                    }
                )
                break
            report.errors.extend(line_errors[:remaining])
            if len(report.errors) >= max_errors:
                report.errors.append(
                    {
                        'loc': '(truncated)',
                        'msg': (
                            f'Error limit reached ({max_errors}). '
                            'Use --max-errors to increase.'
                        ),
                        'type': 'truncated',
                    }
                )
                break

    summary = InstanceFileSummary(
        line_count=report.line_count,
        evaluation_ids=frozenset(evaluation_ids),
        model_ids=frozenset(model_ids),
        content_valid=content_valid,
    )

    if run_semantic_checks:
        if repo_path is None:
            report.valid = False
            report.errors.append(
                {
                    'loc': '(semantic checks)',
                    'msg': 'repo_path is required for repository validation',
                    'type': 'semantic_check_error',
                }
            )
            return report
        if available_files is None:
            available_files = frozenset({repo_path})
        context = ValidationContext(
            repo_path=repo_path,
            available_files=available_files,
            read_repo_file=read_repo_file,
        )
        try:
            semantic_report = run_registered_checks(
                context, file_type='instance', data=summary
            )
            report.errors.extend(semantic_report.errors)
            report.warnings.extend(semantic_report.warnings)
            if semantic_report.errors:
                report.valid = False
        except SemanticCheckError as exc:
            report.valid = False
            report.errors.append(
                {
                    'loc': '(semantic checks)',
                    'msg': str(exc),
                    'type': 'semantic_check_error',
                }
            )

    return report


def validate_file(
    file_path: Path,
    max_errors: int = DEFAULT_MAX_ERRORS,
    *,
    repo_path: str | None = None,
    available_files: Container[str] | None = None,
    read_repo_file: Callable[[str], str] | None = None,
    run_semantic_checks: bool = False,
) -> ValidationReport:
    """Dispatch validation by extension."""
    if file_path.suffix == '.json':
        return validate_aggregate(
            file_path,
            repo_path=repo_path,
            available_files=available_files,
            read_repo_file=read_repo_file,
            run_semantic_checks=run_semantic_checks,
        )
    if file_path.suffix == '.jsonl':
        return validate_instance_file(
            file_path,
            max_errors=max_errors,
            repo_path=repo_path,
            available_files=available_files,
            read_repo_file=read_repo_file,
            run_semantic_checks=run_semantic_checks,
        )

    report = ValidationReport(
        file_path=file_path, valid=False, file_type='unsupported'
    )
    report.errors.append(
        {
            'loc': '(file)',
            'msg': (
                f"Unsupported file extension '{file_path.suffix}'. "
                'Expected .json or .jsonl'
            ),
            'type': 'unsupported_extension',
        }
    )
    return report
