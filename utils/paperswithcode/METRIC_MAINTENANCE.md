# Maintaining metric bounds for the Papers with Code adapter

*Scope: this file is specific to the PwC adapter (`utils/paperswithcode/`). It is
not part of the general Every Eval Ever project — it documents the one piece of
this adapter that needs periodic human/agent judgment: giving every metric a
canonical **min / max / direction** so the emitted `EvaluationLog`s are valid and
comparable.*

If you only ever do one thing here: **run the health check, register the
`RECURRING` bucket, and leave the `BESPOKE` bucket to `--allow-unresolved` until
you have time to read papers.** That keeps data flowing and never ships a wrong
bound. Everything below explains why that is safe and how to do the paper-reads
well when you get to them.

---

## 1. Why this is manual at all

The PwC dump gives us, per metric: the **name**, a `direction` hint
(higher/lower better), a free-text unit/scale hint, an `evaluation_description`,
and a **paper URL**. It does **not** give bounds. But the EEE schema requires a
finite `min_score`/`max_score` for every `continuous` metric (the validator
raises otherwise), and comparability requires a *canonical* scale, not whatever
scale a given leaderboard happened to report in.

So bounds have to come from somewhere deterministic. That "somewhere" is the
**eval-card-registry** (`seed/metrics.yaml`). Populate it once from cited
definitions and every future run resolves names → bounds by static YAML lookup —
**no LLM at resolution time.** The registry is the durable, non-LLM source of
truth; an LLM/agent is only ever used *once*, at registration time, to read a
paper — and its answer is then frozen as a citation in the entry.

```
PwC dump ──► adapter ──► resolve(name) ──► registry_metrics.json (snapshot of
                                            eval-card-registry seed/metrics.yaml)
                                          └► bounds + direction, deterministically
```

`registry_metrics.json` is a **vendored snapshot**; regenerate it with
`refresh_metric_snapshot.py` after every registry change (see §5).

---

## 2. The health check (start here every time)

```bash
# from the repo root, with every_eval_ever importable:
PYTHONPATH=$(pwd) python utils/paperswithcode/adapter.py \
    --dump <dump.dump> --all --best-effort --output-dir /tmp/eee-pwc-check
```

`--best-effort` emits everything and exits 0, but **still prints the full
imperfection report to stderr**. The unresolved section is now **triaged into
three buckets** for you:

- **RECURRING** — the name matches a known standard family. A ~60-second registry
  add with the family's bound/direction (§3). Do these.
- **BESPOKE** — no family match; needs a read of the defining paper (§4). Do these
  when you have time; `--allow-unresolved` covers them meanwhile.
- **AMBIGUOUS** — the name matches *two* canonical ids. This is a duplicate
  alias/display_name in the registry, **not** a missing metric. Fix the collision
  in `seed/metrics.yaml` and refresh.

The triage uses `classify_metric_family()` in `adapter.py` — the same family
taxonomy that seeded the recurring metrics. It is a **hint, not an auto-rule**
(see the warning in §4).

---

## 3. RECURRING bucket — register a standard-family metric

These follow from the family definition, so you can register them without
reading a paper — but **confirm the bound matches the family** (a name can lie;
see §4). Family → canonical bound/direction:

| family       | min | max  | lower_is_better | typical names |
|--------------|-----|------|-----------------|---------------|
| `rate`       | 0.0 | 1.0  | false           | accuracy, F1, AP/mAP, AUROC/AUPRC, recall@k, IoU, R@k IoU=x, success rate, OLS |
| `pose-error` | 0.0 | null | true            | MPJPE, PVE/MVE, N-MPJPE, PA-PVE, MRPE, end-point error (mm) |
| `dist-error` | 0.0 | null | true            | MSE, RMSE, MAE, L1/L2 distance (m/mm) |
| `spec-loss`  | 0.0 | null | true            | Mel Loss, STFT distance, F0-RMSE |
| `psnr`       | 0.0 | null | false           | PSNR (dB) |
| `pesq`       | -0.5| 4.5  | false           | PESQ, PESQ-NB, PESQ-WB |
| `mos`        | 1.0 | 5.0  | false           | MOS, DNSMOS, PLCMOS, UTMOS, ViSQOL |
| `stoi`       | 0.0 | 1.0  | false           | STOI, ESTOI |
| `mcd`        | 0.0 | null | true            | MCD, mel-cepstral distortion |
| `gen-dist`   | 0.0 | null | true            | FID, FVD, rFVD, KID |
| `bitrate`    | 0.0 | null | true            | bpp, bpsp, bits-per-* |
| `bd`         | null| null | true            | BD-Rate, Bjontegaard-delta (signed %) |

`null` max = genuinely unbounded above (the adapter serializes it as `Infinity`,
see [every_eval_ever#207]). `null`/`null` = signed & unbounded.

**`rate` and the percent question.** Register `rate` metrics as **[0, 1]**. Many
leaderboards report them as percent (0–100); the adapter reconciles that *per
`(metric, dataset)` group* from the group median and rescales percent → proportion
automatically (`reconcile_scale`). You do **not** register a second [0,100]
version. The only time you register a `> 1` max (e.g. `[0, 100]`) is a metric that
is *intrinsically* on that scale and never a proportion (see §6).

### Entry template

Append to the appropriate `# --- family: ... ---` group in
`eval-card-registry/seed/metrics.yaml`:

```yaml
- id: n-mpjpe                       # kebab-case, unique; the adapter also matches
  display_name: 'N-MPJPE'           # by normalized alias (case/-/space-insensitive)
  aliases:
  - 'N-MPJPE'                       # every surface form the dump uses
  score_type: continuous
  lower_is_better: true
  min_score: 0.0
  max_score: null                   # unbounded above
  metadata: '{"kind": "real", "confidence": "high", "family": "pose-error", "source": "3D joint position error (mm); [0,inf), lower better", "provenance": "paperswithcode-adapter"}'
  review_status: draft              # flip to reviewed once a human has verified
```

Then **refresh + re-run** (§5). Done.

---

## 4. BESPOKE bucket — read the paper

A bespoke metric is a paper-specific composite (e.g. `WorldScore-Dynamic`,
`Driving Score`, `EPDMS`, `PIE-Bench Background LPIPS`). There is no family
default; you must read its defining paper. This is the part that needs judgment.

> ### ⚠️ Why you cannot infer bounds from the name
> Name inference is unsafe **for bounds**, even when it is fine for triage:
> - `PIE-Bench Background LPIPS` — LPIPS is definitionally [0,1], but PIE-Bench
>   reports it **×10³**, so observed values are 62–304. The honest registry bound
>   is **[0, 1000]** (the scale the dump uses), not [0,1]. Only the paper tells
>   you the ×10³.
> - `BD-Rate (PSNR RGB)` — the name contains "PSNR", but it is a Bjontegaard
>   **rate**: signed, unbounded, lower-better — not a [0,∞) higher-better PSNR.
> - `S-BERT`, `F0-CORR`, cosine similarities — bounded **[-1, 1]**, not [0,1],
>   even though observed values sit in the positive range.
>
> This is exactly why the family table in §3 is a *hint* and every bound is
> **cross-checked against observed data** (§4.3) before it is trusted.

### 4.1 Find the paper

Each metric's paper URL travels with the data. In the emitted logs it is under
`score_details.details.paper_arxiv_url` / `source_url` /
`external_source_url`, and `source_data.additional_details.paper_url`. In the raw
dump it is on the evaluation/result row. A PwC `/paper/<id>` slug maps to
`arxiv.org/abs/<id>`; numeric PwC ids may need a PwC-page or title search.

### 4.2 Deduce the bound shape

Read for the metric's **definition** and decide the shape:

| the paper says… | register as |
|---|---|
| a probability / rate / fraction / normalized-to-[0,1] score | `[0, 1]`, higher better (unless it's an error) |
| a percentage the paper reports on 0–100 and never as a fraction | `[0, 100]` |
| a cosine / correlation | `[-1, 1]` |
| an error / distance / divergence, ≥ 0, no ceiling | `[0, null]`, lower better |
| a score in std-dev units, or a signed delta (can be negative, no ceiling) | `[null, null]` |
| a rubric score 1–N (MOS, GPT-judge 1–10) | `[1, N]` (note the floor is the rubric min, not 0) |
| bounded on a stated interval | that interval |

**Direction:** trust the paper's arrow, not the PwC `direction` hint (they
disagreed in real cases — e.g. RealIR's `LPS` is reported lower-better despite a
"similarity" name).

**If the paper is not enough → `null`.** Do **not** guess a bound for a shared
registry others depend on. Register **name-only with null bounds** (keeps the
name/direction/alias resolvable; the adapter emits `Infinity` bounds for it), or
leave it unresolved. Either is honest; a wrong bound is not. Record the
uncertainty in `metadata.confidence` (`low`) and `review_status: draft`.

### 4.3 The observed-range cross-check (do this for EVERY bespoke bound)

Before trusting a paper-claimed finite bound, confirm the dump's observed value
range fits it — this is a deterministic self-audit that catches scale mistakes:

```
observed [omin, omax] fits [lo, hi]                     → OK, use it
does not fit, but [omin/100, omax/100] fits AND hi<=1    → OK (adapter rescales percent→proportion)
does not fit either way                                  → SCALE MISMATCH: re-read the paper for
                                                            a ×10ⁿ reporting convention (LPIPS ×10³),
                                                            or a signed/percent scale you missed.
                                                            Register on the scale the dump uses.
```

This is what caught the 4 mis-bounded `BD-Rate` variants (observed negatives vs a
claimed `[0, ∞)`) and confirmed the PIE-Bench ×10³ scales. `consolidate_bounds`
implements it; the rule above is all you need to redo it by hand.

### 4.4 Doing many at once (how the initial 64 were done)

For a large first pass, batch it: cluster metrics by paper, hand each cluster to
an agent with a payload of `{metric, benchmark_datasets, paper_url, pwc_direction,
observed_range, description}`, and ask for `{min, max, lower_is_better,
scale_note, evidence, source_url, confidence}` per metric with **direct paper
quotes** as evidence. Then run the observed-range cross-check over all results,
demote any that fail, and append the survivors to `metrics.yaml`. This is a
one-time cost; the results are frozen as citations. Agents are for the *read*,
never for the runtime resolution.

### 4.5 Verifying with a model (the minimal path)

If all you can do is "add a metric and verify with a model": paste the metric
name + its paper URL + its observed range, ask the model to find the definition
and state the bound and direction with a quote, **then apply the §4.3
cross-check** against the observed range yourself. The cross-check is the
guardrail that makes a model's answer safe to commit — never skip it.

---

## 5. Refresh + re-run (after ANY registry change)

```bash
python utils/paperswithcode/refresh_metric_snapshot.py \
    --seed <path-to>/eval-card-registry/seed/metrics.yaml
# writes registry_metrics.json + records the registry git revision in _meta
```
Commit the registry change first so the snapshot pins a clean (non-`-dirty`)
revision. Then re-run the health check (§2); the metrics you registered should
move out of the unresolved report. A strict run (no `--best-effort`) should now
exit 0 for them.

---

## 6. Registry scale conventions (cheat-sheet)

Confirmed from the existing registry + adapter behavior:

- proportions / accuracy / recall / AP / AUROC / F1 / IoU → **[0, 1]** (percent
  boards auto-rescaled by group median; do not add a [0,100] twin)
- intrinsic percentages never reported as fractions → **[0, 100]**
- unbounded errors/distances (PSNR, MPJPE, MSE, FID, MCD, bitrate) →
  **max = null**, `lower_is_better` per whether up or down is good
- signed / unbounded both ways (BD-Rate, NSS, score deltas) → **min = max = null**
- MOS-type → **[1, 5]**; GPT-judge rubric → **[1, 10]** (or 0–100 if scaled)
- correlations / cosines → **[-1, 1]**
- aesthetic predictors → **[0, 10]**; angular error → **[0, 180]**

Reminder: the adapter only auto-rescales percent→proportion when the canonical
`max <= 1`. If you register a `max > 1`, values are range-checked **as-is** (no
rescale) and anything outside is flagged `scale_anomaly` — so pick the max that
matches the scale the dump actually reports for that metric.

---

## 7. What "good" looks like

- Every new entry cites its basis in `metadata.source` (+ `metadata.paper` for
  bespoke) and starts at `review_status: draft`.
- No entry asserts a bound the observed data contradicts (run §4.3).
- Unknowns are `null`, not guessed.
- The snapshot is refreshed and its `_meta.registry_revision` is clean.
- A strict `--all` run's remaining unresolved set is only genuinely-new or
  genuinely-unknowable metrics — everything else resolved to a cited bound.

[every_eval_ever#207]: https://github.com/evaleval/every_eval_ever/pull/207
