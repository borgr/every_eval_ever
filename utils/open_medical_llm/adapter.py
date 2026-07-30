#!/usr/bin/env python3
"""Convert the Open Medical-LLM Leaderboard results into Every Eval Ever aggregate logs.

Data source (HuggingFace dataset): openlifescienceai/results
  Layout: <developer>/<model>/results_*.json  (lm-evaluation-harness output format)
  Backs the Space: openlifescienceai/open_medical_llm_leaderboard

One EvaluationLog per model (developer/model), with one EvaluationResult per medical
benchmark (accuracy, proportion 0-1, higher is better). Aggregates only; no per-item data.

Run from the EEE repo dir:
    uv run python -m utils.open_medical_llm.adapter --output-dir /tmp/eee-omll [--limit N]
    uv run python -m every_eval_ever validate /tmp/eee-omll
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone

import requests

from every_eval_ever.eval_types import (
    EvalLibrary,
    EvaluationLog,
    EvaluationResult,
    EvaluatorRelationship,
    MetricConfig,
    ModelInfo,
    ScoreDetails,
    ScoreType,
    SourceDataHf,
    SourceMetadata,
    StandardError,
    Uncertainty,
)
from every_eval_ever.helpers import (
    SCHEMA_VERSION,
    save_evaluation_log,
)

REPO = "openlifescienceai/results"
RESOLVE = "https://huggingface.co/datasets/openlifescienceai/results/resolve/main/"
TREE = "https://huggingface.co/api/datasets/openlifescienceai/results/tree/main?recursive=true"
LEADERBOARD_SPACE = "https://huggingface.co/spaces/openlifescienceai/open_medical_llm_leaderboard"
SRC = "open-medical-llm-leaderboard"

# Hosted eval-card-registry resolver (public HF Space, no auth). Maps a raw HF
# ``developer/model`` id to the shared canonical id. See resolve_model_id.
RESOLVER_URL = "https://evaleval-entity-registry.hf.space/api/v1/resolve"
# below this the resolver's alias is treated as unverified (flag for review):
RESOLVE_CONFIDENCE_FLOOR = 0.9

# task name -> (human display name, verified HF dataset repo)
TASKS = {
    "medmcqa": ("MedMCQA", "openlifescienceai/medmcqa"),
    "medqa_4options": ("MedQA (USMLE, 4 options)", "openlifescienceai/MedQA-USMLE-4-options-hf"),
    "pubmedqa": ("PubMedQA", "openlifescienceai/pubmedqa"),
    "mmlu_anatomy": ("MMLU: Anatomy", "openlifescienceai/mmlu_anatomy"),
    "mmlu_clinical_knowledge": ("MMLU: Clinical Knowledge", "openlifescienceai/mmlu_clinical_knowledge"),
    "mmlu_college_biology": ("MMLU: College Biology", "openlifescienceai/mmlu_college_biology"),
    "mmlu_college_medicine": ("MMLU: College Medicine", "openlifescienceai/mmlu_college_medicine"),
    "mmlu_medical_genetics": ("MMLU: Medical Genetics", "openlifescienceai/mmlu_medical_genetics"),
    "mmlu_professional_medicine": ("MMLU: Professional Medicine", "openlifescienceai/mmlu_professional_medicine"),
}

# Capture optional fractional seconds so two runs of the same model in the same
# whole second stay distinguishable (see parse_ts / make_log ts_token). Also allow
# an ISO 'T' separator alongside the leaderboard's space/underscore form.
TS_RE = re.compile(
    r"(\d{4})-(\d{2})-(\d{2})[ _T](\d{2})[:_-](\d{2})[:_-](\d{2})(?:[.,](\d{1,6}))?"
)


def stringify(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (dict, list)):
        return json.dumps(v, sort_keys=True, separators=(",", ":"))
    return str(v)


def clean_details(d: dict) -> dict:
    return {k: stringify(v) for k, v in d.items() if v is not None}


def parse_ts(path: str):
    m = TS_RE.search(path)
    if not m:
        return None
    y, mo, da, h, mi, s = (int(g) for g in m.groups()[:6])
    frac = m.group(7)
    micros = int(frac.ljust(6, "0")[:6]) if frac else 0  # right-pad ms->us, cap at us
    try:
        return datetime(y, mo, da, h, mi, s, micros, tzinfo=timezone.utc)
    except ValueError:
        return None


def fetch_json(path: str) -> dict:
    url = RESOLVE + urllib.parse.quote(path)
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read())


def _next_link(link_header: str | None) -> str | None:
    """Extract the ``rel="next"`` URL from an RFC-5988 ``Link`` header (HF tree
    pagination), or ``None`` when there is no next page."""
    if not link_header:
        return None
    for part in link_header.split(","):
        seg = part.strip()
        if 'rel="next"' not in seg:
            continue
        lt, gt = seg.find("<"), seg.find(">")
        if lt != -1 and gt != -1:
            return seg[lt + 1 : gt]
    return None


def _iter_tree_pages(url: str):
    """Yield tree entries across ALL pages. The HF tree API caps a page at ~1000
    entries and points at the next via a ``Link: <...>; rel="next"`` header — a
    single unpaginated GET silently truncates large repos."""
    while url:
        req = urllib.request.Request(url, headers={"User-Agent": "eee-omll-adapter"})
        with urllib.request.urlopen(req, timeout=60) as r:
            yield from json.loads(r.read())
            url = _next_link(r.headers.get("Link"))


def list_result_files() -> list[str]:
    out = []
    for x in _iter_tree_pages(TREE):
        if x.get("type") != "file":
            continue
        p = x["path"]
        segs = p.split("/")
        # skip hidden dirs (e.g. .ipynb_checkpoints) and jupyter checkpoint files
        if any(s.startswith(".") for s in segs) or segs[-1].endswith("-checkpoint.json"):
            continue
        if p.endswith(".json") and segs[-1].startswith("results_"):
            out.append(p)
    return out


def resolve_model_id(raw_repo: str, *, enabled: bool = True, timeout: float = 15.0) -> tuple[str, dict]:
    """Canonicalize an HF ``developer/model`` id via the hosted eval-card-registry
    resolver. Returns ``(model_id, provenance)``.

    Design (mirrors the skill's reference/registry.md + fields.md model_info):
    - DEFAULT resolves against the registry so ``model_info.id`` is the shared
      canonical JOIN key. Uses ``requests`` (already a dep) — no new dependency,
      so ``--no-registry-resolve`` is purely a speed/offline/determinism opt-out.
    - ``evaluation_id`` is deliberately NOT keyed on this value (see make_log):
      the registry may later re-map a freshly auto-created draft, and a moving
      canonical id would break re-ingest idempotency. Resolved id = join key,
      raw path = record identity.
    - The resolver's last strategy is *auto-create draft*, so ``created_new`` /
      ``review_status`` / ``confidence`` are surfaced; main() summarizes the ones
      that need registry review (the skill's "creating a new canonical id" is an
      operator-policy call — here we make it visible rather than block a batch).
    - Never fatal: opt-out or any network error falls back to the path id and
      records why (``offline`` / ``unreachable``).
    """
    if not enabled:
        return raw_repo, {"model_id_resolution": "offline"}
    try:
        resp = requests.post(
            RESOLVER_URL,
            json={"raw_value": raw_repo, "entity_type": "model"},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:  # noqa: BLE001 — resolution is best-effort, never fatal
        return raw_repo, {"model_id_resolution": "unreachable",
                          "model_id_resolution_error": str(e)[:200]}
    canonical = data.get("canonical_id") or raw_repo
    return canonical, {
        "model_id_resolution": "registry",
        "model_id_resolution_strategy": data.get("strategy"),
        "model_id_resolution_confidence": data.get("confidence"),
        "model_id_created_new": data.get("created_new"),
        "model_id_review_status": data.get("review_status"),
    }


def _needs_registry_review(prov: dict | None) -> bool:
    """True when a resolved id is not a confident, already-reviewed canonical:
    unreachable resolver, a freshly auto-created draft, a non-``reviewed`` status,
    or confidence below the floor. (``offline`` is reported once, not per model.)"""
    if not prov:
        return False
    if prov.get("model_id_resolution") == "unreachable":
        return True
    if prov.get("model_id_created_new"):
        return True
    status = prov.get("model_id_review_status")
    if status not in (None, "reviewed"):
        return True
    conf = prov.get("model_id_resolution_confidence")
    return isinstance(conf, (int, float)) and conf < RESOLVE_CONFIDENCE_FLOOR


def latest_per_model(paths: list[str]) -> tuple[dict[str, str], list[str]]:
    """Group 3-segment ``developer/model/results_*.json`` paths, latest file per model.

    Root-level 2-segment baselines (e.g. ``GPT-4/results_*.json``) are returned
    separately and skipped: they are hand-curated closed-model paper numbers
    (bare ``acc`` only, no ``acc_stderr``/``model_args``/``bootstrap_iters``),
    NOT lm-evaluation-harness runs, so their provenance differs from the rest.
    """
    by_model: dict[str, list[str]] = defaultdict(list)
    baselines: list[str] = []
    for p in paths:
        parts = p.split("/")
        if len(parts) < 3:
            baselines.append(p)
            continue
        by_model["/".join(parts[:2])].append(p)
    chosen = {}
    for model, files in by_model.items():
        files.sort(key=lambda p: (parse_ts(p) or datetime.min.replace(tzinfo=timezone.utc), p))
        chosen[model] = files[-1]
    return chosen, baselines


def make_result(task: str, metrics: dict, eval_ts_iso: str | None) -> EvaluationResult | None:
    acc = metrics.get("acc,none")
    if acc is None:
        return None
    display, hf_repo = TASKS[task]

    stderr = metrics.get("acc_stderr,none")
    uncertainty = None
    if stderr is not None:
        uncertainty = Uncertainty(standard_error=StandardError(value=float(stderr)))

    score_details = ScoreDetails(
        score=float(acc),
        details=clean_details(
            {
                "raw_metric_key": "acc,none",
                "acc_norm": metrics.get("acc_norm,none"),
                "acc_norm_stderr": metrics.get("acc_norm_stderr,none"),
                "harness_alias": metrics.get("alias"),
            }
        ),
        uncertainty=uncertainty,
    )

    return EvaluationResult(
        evaluation_result_id=f"{SRC}.{task}",
        evaluation_name=f"{SRC}.{task}",
        evaluation_timestamp=eval_ts_iso,
        source_data=SourceDataHf(
            dataset_name=display,
            source_type="hf_dataset",
            hf_repo=hf_repo,
        ),
        metric_config=MetricConfig(
            evaluation_description=(
                f"Accuracy on the {display} medical QA benchmark as reported by the "
                "Open Medical-LLM Leaderboard."
            ),
            metric_id=f"{SRC}.{task}.accuracy",
            metric_name="accuracy",
            metric_kind="accuracy",
            metric_unit="proportion",
            lower_is_better=False,
            score_type=ScoreType.continuous,
            min_score=0.0,
            max_score=1.0,
        ),
        score_details=score_details,
    )


def make_log(
    model_repo: str,
    obj: dict,
    path: str,
    retrieved_ts: str,
    *,
    model_id: str | None = None,
    resolution_details: dict | None = None,
) -> tuple[EvaluationLog, str, str] | None:
    """Build one aggregate log for ``developer/model``.

    ``model_id`` is the registry-canonical id for ``model_info.id`` (the join
    key); pass ``None`` for path-mode (id == source repo). ``evaluation_id`` is
    ALWAYS keyed on the raw source repo, never on ``model_id`` — see
    resolve_model_id for why (idempotency vs. a movable canonical id). Offline
    unit tests call this directly without a resolver, so it never touches the
    network.
    """
    developer, model = model_repo.split("/", 1)
    config = obj.get("config", {}) or {}
    results = obj.get("results", {}) or {}

    ev_results = []
    for task in TASKS:
        md = results.get(task)
        if isinstance(md, dict):
            r = make_result(task, md, None)  # per-result ts filled below
            if r is not None:
                ev_results.append(r)
    if not ev_results:
        return None

    eval_dt = parse_ts(path)
    eval_ts_iso = eval_dt.isoformat() if eval_dt else None
    for r in ev_results:
        r.evaluation_timestamp = eval_ts_iso

    if eval_dt:
        base = int(eval_dt.timestamp())
        # keep sub-second precision so two runs in the same whole second differ
        ts_token = f"{base}.{eval_dt.microsecond:06d}" if eval_dt.microsecond else str(base)
    else:
        # No parseable timestamp in the filename: derive a STABLE token from the
        # result file path (never `now`), so evaluation_id stays idempotent across
        # re-runs. evaluation_timestamp is left None (the run time is unknown).
        ts_token = re.sub(r"[^0-9A-Za-z._-]+", "-", path.rsplit("/", 1)[-1].removesuffix(".json"))

    # model_info.id = registry-canonical join key (or the source repo in path-mode).
    # evaluation_id stays keyed on the RAW repo below, so re-ingest is idempotent
    # even if the registry later re-maps a draft canonical id.
    resolved_id = model_id or model_repo
    raw_slug = model_repo.replace("/", "_")

    md_details: dict = {
        "model_sha": config.get("model_sha"),
        "model_dtype": config.get("model_dtype"),
        "model_args": config.get("model_args"),
        "model_num_parameters": config.get("model_num_parameters"),
        "model_revision": config.get("model_revision"),
    }
    if resolution_details:
        md_details.update(resolution_details)
    if resolved_id != model_repo:
        md_details["source_model_repo"] = model_repo  # keep the raw->canonical mapping visible
    config_name = config.get("model_name")
    if config_name and config_name != model_repo:
        # config's model_name and the repo path disagree (either can be the typo);
        # id follows the chosen source consistently — record BOTH so the maintainer
        # can adjudicate rather than trusting one silently.
        md_details["config_model_name"] = config_name
        md_details["name_path_divergence"] = True

    model_info = ModelInfo(
        name=config_name or model_repo,
        id=resolved_id,
        developer=developer,
        additional_details=clean_details(md_details),
    )

    log = EvaluationLog(
        schema_version=SCHEMA_VERSION,
        evaluation_id=f"{SRC}/{raw_slug}/{ts_token}",
        evaluation_timestamp=eval_ts_iso,
        retrieved_timestamp=retrieved_ts,
        source_metadata=SourceMetadata(
            source_name="Open Medical-LLM Leaderboard",
            source_type="documentation",
            source_organization_name="Open Life Science AI",
            source_organization_url=LEADERBOARD_SPACE,
            evaluator_relationship=EvaluatorRelationship.third_party,
            additional_details={
                "source_role": "aggregator",
                "results_dataset": REPO,
                "source_result_file": path,
            },
        ),
        eval_library=EvalLibrary(
            name="lm-evaluation-harness",
            version="unknown",
            additional_details={
                "inferred_from": "result format (acc,none / acc_stderr,none / bootstrap_iters / fewshot_seed)"
            },
        ),
        model_info=model_info,
        evaluation_results=sorted(ev_results, key=lambda r: r.evaluation_name),
    )
    return log, developer, model


def main() -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="data/open-medical-llm")
    ap.add_argument("--limit", type=int, default=None, help="Process only the first N models.")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--no-registry-resolve",
        action="store_true",
        help="Skip the eval-card-registry lookup and use the path-derived HF id as "
             "model_info.id (faster / offline / deterministic, but NOT canonicalized).",
    )
    args = ap.parse_args()
    resolve_enabled = not args.no_registry_resolve

    retrieved_ts = str(time.time())
    paths = list_result_files()
    chosen, baselines = latest_per_model(paths)
    models = sorted(chosen)
    if args.limit:
        models = models[: args.limit]
    print(f"Models to process: {len(models)}")
    print(
        f"Skipped {len(baselines)} hand-curated baseline entries (different provenance): "
        + ", ".join(sorted(p.split('/')[0] for p in baselines))
    )
    if not resolve_enabled:
        print("  NOTE: --no-registry-resolve set; model_info.id is path-derived and NOT registry-verified.")

    def worker(model_repo: str):
        # make_log is INSIDE the try: a malformed record must not escape ex.map()
        # and abort the whole run — it becomes a per-model error instead.
        try:
            obj = fetch_json(chosen[model_repo])
            model_id, prov = resolve_model_id(model_repo, enabled=resolve_enabled)
            built = make_log(model_repo, obj, chosen[model_repo], retrieved_ts,
                             model_id=model_id, resolution_details=prov)
        except Exception as e:  # noqa: BLE001
            return ("ERR", model_repo, str(e), None)
        if built is None:
            return ("SKIP", model_repo, "no medical results", None)
        return ("OK", model_repo, built, prov)

    written = errors = skipped = 0
    flagged: list[tuple[str, dict]] = []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for status, model_repo, payload, prov in ex.map(worker, models):
            if status == "OK":
                log, developer, model = payload
                save_evaluation_log(log, args.output_dir, developer, model)
                written += 1
                if resolve_enabled and _needs_registry_review(prov):
                    flagged.append((model_repo, prov))
            elif status == "SKIP":
                skipped += 1
            else:
                errors += 1
                print(f"  ERROR {model_repo}: {payload}")
    print(f"Wrote {written} logs; skipped {skipped}; errors {errors}. -> {args.output_dir}")
    if flagged:
        print(f"\n  {len(flagged)} model id(s) need registry review "
              "(unresolved / auto-created draft / low-confidence / unreviewed):")
        for mr, prov in flagged:
            print(f"    - {mr}: resolution={prov.get('model_id_resolution')} "
                  f"strategy={prov.get('model_id_resolution_strategy')} "
                  f"confidence={prov.get('model_id_resolution_confidence')} "
                  f"created_new={prov.get('model_id_created_new')} "
                  f"review_status={prov.get('model_id_review_status')}")
        print("  -> record these in the PR decision log and prepare a registry alias PR.")
    return {"written": written, "errors": errors, "skipped": skipped, "flagged": len(flagged)}


if __name__ == "__main__":            # run:  uv run python -m utils.open_medical_llm.adapter
    summary = main()
    sys.exit(1 if summary["errors"] else 0)   # non-zero on partial failure (was always 0)
