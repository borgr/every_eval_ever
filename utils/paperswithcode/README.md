# Papers with Code Adapter

Converts [Papers with Code](https://paperswithcode.com) leaderboard/evaluation
results into Every Eval Ever aggregate `EvaluationLog` JSON files.

## Data source

Nightly PostgreSQL backups of the PwC database, published to the HF **bucket**
`huggingface/paperswithcode-backups` under `postgres/*.dump` (pg_dump custom format,
`-Fc`). Each daily dump is ~180–210 MB; the bucket holds ~30 days (~6 GB total).

Dumps are read with [`pgdumplib`](https://pypi.org/project/pgdumplib/) — a
pure-Python reader, so **no PostgreSQL server or `pg_restore` is required**:

```bash
pip install pgdumplib
```

> Note: reading the bucket requires `huggingface_hub>=1.0` (the `HfApi.bucket_*`
> API). The repo's pinned range is `>=0.36,<1.0`; install a 1.x build in the
> environment you run this adapter from.

### Tables used

| Table | Role |
|-------|------|
| `evaluations` | one row per (paper, task, dataset, model) leaderboard entry; `metrics` jsonb = `{metric_name: value}` |
| `datasets` | the benchmark the eval ran on → `source_data` |
| `tasks` | the task/category the benchmark belongs to → part of `evaluation_name` |
| `metrics` | metric definitions incl. `direction` (lower/higher_is_better) → `lower_is_better` |
| `papers` | provenance (arXiv id / source url) |

## Shape decisions

- **`source_type = documentation`** — PwC aggregates *reported* numbers (often
  re-reported from external leaderboards); this adapter does not re-run anything.
- **Aggregate `.json` only** — PwC has no per-item data, so no `_samples.jsonl`.
- **One log per model** — evaluations are grouped by canonical model id; each
  `evaluation_results[]` entry is one (evaluation row × metric-in-jsonb) pair.
- **`evaluation_id`** is keyed on `model_id` + dump date → stable/idempotent per
  dump, never on `now`.
- **Idempotent output** — a re-run over the same dump is byte-stable:
  `retrieved_timestamp` is pinned to the dump date (not wall-clock `time.time()`),
  and the output dir is **replaced** (wiped, after the fail-closed gate) each run
  so the uuid4-named files don't accumulate duplicates across runs.
- **Metric bounds/direction come from the registry, not invented** — see
  "Metric resolution" below.

## Metric resolution (three tiers, fail-closed)

`continuous` metrics require `min_score`/`max_score`, which are the metric's
*defined* range — so they're sourced from the eval-card-registry's canonical
metric entries (vendored offline in `registry_metrics.json`, refreshed via
`refresh_metric_snapshot.py`), never guessed per-adapter:

1. **Resolved** — metric matches a canonical entry → use its
   `min_score`/`max_score`/`lower_is_better`/`score_type` and canonical `metric_id`.
   Matching is **exact-first**: a case-insensitive match on id/display_name/alias
   wins before the lossy normalized fallback (case + separators dropped), so
   distinct-but-similar names resolve to their *own* id (`CLIP-IQA` → `clip-iqa`,
   `CLIPIQA+` → `clipiqa-plus`) instead of whichever the index saw first. The hit
   records `match_tier` (`exact`/`normalized`), the entry's `review_status` +
   `confidence`/`kind`, and the `bound_registry_revision` (the exact registry
   commit the bound came from) — surfaced, not used to reject a still-`draft` entry.
2. **Unresolved (default: FAIL CLOSED)** — a metric not in the snapshot **or an
   ambiguous name collision** (one spelling mapping to >1 canonical id — a
   duplicate alias in the registry) aborts the run non-zero, naming each metric,
   distinguishing *unknown* from *AMBIGUOUS*, and pointing at the registry's
   `registry-entity-aliases` skill. CI runs this default, so a new/ambiguous PwC
   metric can never silently ship un-vetted or mis-attributed bounds.
3. **`--allow-unresolved` (opt-out)** — emit unresolved metrics (unknown *and*
   ambiguous) with observed-range bounds (`bound_source=observed_unresolved`,
   `collision_candidates` listed for ambiguous ones) + a warning summary, for
   humans doing exploratory runs.

Three reconciliations keep resolved records on the canonical scale:
- **Direction** — `lower_is_better` is required by the schema (non-nullable), so it
  is resolved by a priority chain, recorded in `score_details.details` as
  `direction_source`: the registry entry's direction wins; else the PwC `metrics`
  table's own `direction` column (`lower/higher_is_better`) is used
  (`direction_source=pwc_source`); else it defaults to `False` and is **flagged**
  `direction_source=unknown` (a gated imperfection — see run modes). A registry
  entry with `lower_is_better: null` (direction genuinely context-dependent) is
  honoured, not overwritten: the PwC column fills it in per row.
- **Scale (group-level)** — PwC reports proportion metrics (canonical `[0,1]`) as
  percent (0–100), inconsistently even *within* one leaderboard. The reporting
  scale is a property of the whole `(metric, dataset)` group, so it is decided
  **once per group from that group's median**, not per score. A score is then
  rescaled by the group's factor (e.g. `87.3 → 0.873`), recording `raw_value`,
  `canonical_rescale_factor`, and `rescale_basis` (`group_median` / `single_score`)
  in `score_details.details`. This is robust both ways vs. per-score inference: a
  lone in-range value in an otherwise-percent board (a `1.0` that means 1%) is
  rescaled to match the group, and a lone out-of-range value in an otherwise-
  proportion board (a mis-entered `95`) is **flagged** `scale_anomaly` and kept,
  not silently divided. A group centred above 100, or a score still outside the
  canonical range after the group scale, is flagged rather than guessed at.
- **Unbounded** — metrics with a `null` bound in the registry (PSNR, AbsRel,
  Chamfer, …) are emitted with `±inf`, which serializes to the JSON string
  `"Infinity"`/`"-Infinity"` per
  [every_eval_ever#207](https://github.com/evaleval/every_eval_ever/pull/207) — valid
  JSON that reads back to a float. `null` means "not provided", never unbounded. The
  run prints a `NOTE` listing which metrics were emitted with unbounded bounds.

> **Depends on #207.** The `"Infinity"` bound serialization requires the field
> serializer added in every_eval_ever#207. This branch is stacked on it; if #207
> changes or is rejected, the unbounded-emit path needs revisiting (the resolver is
> the single choke-point). Rebase onto `main` once #207 merges.

## Run modes (strict vs best-effort)

Every run **always prints a full imperfection report** (to stderr) covering three
classes: *unresolved* metrics, *unknown-direction* metrics, and *scale anomalies*.
The modes decide whether to abort, never whether to report:

- **strict (default)** — abort non-zero, before writing anything, if *any*
  imperfection is present. This is the CI signal: a clean run means every emitted
  record has a registry-sourced bound, a known direction, and an in-range score.
- **`--best-effort`** — emit as much data as possible; every imperfection stays
  flagged in the output (`bound_source`, `direction_source`, `scale_anomaly`) and
  the run exits 0. For humans who want maximum coverage from one dump.
- **`--allow-unresolved`** — a narrow relaxation of strict: tolerate *only* the
  unresolved class (observed-range bounds), still failing on unknown direction or
  scale anomalies.

## Usage

From a dump already on disk (no network):

```bash
uv run python -m utils.paperswithcode.adapter \
  --dump /tmp/pwc-raw/paperswithcode_hf_20260716_031511.dump \
  --dataset-slug eth3d-relative --dataset-slug re10k-2-view \
  --output-dir /tmp/eee-pwc
```

Download the latest dump from the bucket and convert a small default sample:

```bash
uv run python -m utils.paperswithcode.adapter --output-dir data/paperswithcode
```

Convert everything (large):

```bash
uv run python -m utils.paperswithcode.adapter --dump <path> --all \
  --output-dir data/paperswithcode
```

Validate:

```bash
python -m every_eval_ever validate /tmp/eee-pwc
```

## Canonical ids (registry)

Model ids use the HF `developer/model` form (taken verbatim from `hf_model_url`
when present, else guessed from the model name). Benchmark ids are the dotted
`paperswithcode.<task-slug>.<dataset-slug>`. Effort/mode tiers in PwC model names
(e.g. `GPT-5.5 Pro (xhigh)`) are **not** stripped here and non-LLM developers fall
back to `unknown`; collapsing tiers and resolving/aliasing ids against the
`eval-card-registry` is a separate PR to that repo (see its `CONTRIBUTING.md`).

## Options

Run `--help` for the full list: `--dump`, `--bucket`, `--remote-path`,
`--raw-dir`, `--dataset-slug` (repeatable), `--all`, `--limit`,
`--allow-unresolved`, `--best-effort`, `--output-dir`.
