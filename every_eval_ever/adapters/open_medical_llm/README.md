# Open Medical-LLM Leaderboard adapter

Converts the [Open Medical-LLM Leaderboard](https://huggingface.co/spaces/openlifescienceai/open_medical_llm_leaderboard)
results into Every Eval Ever aggregate logs.

- **Source (data):** HF dataset [`openlifescienceai/results`](https://huggingface.co/datasets/openlifescienceai/results),
  laid out as `<developer>/<model>/results_*.json` in lm-evaluation-harness output format.
- **Grain:** one `EvaluationLog` per model (`developer/model`), with one
  `EvaluationResult` per medical benchmark. Aggregates only — no per-item data.
- **Benchmarks (9):** MedMCQA, MedQA (USMLE 4-options), PubMedQA, and six MMLU
  medical subjects (anatomy, clinical knowledge, college biology, college
  medicine, medical genetics, professional medicine). Each result's
  `source_data` points at that benchmark's own HF dataset repo.
- **Metric:** `acc,none` → accuracy, `continuous` in `[0, 1]`, higher is better;
  `acc_stderr,none` → `uncertainty.standard_error` when present.
- **Provenance:** the leaderboard re-hosts numbers it did not itself produce, so
  `source_type="documentation"` and `evaluator_relationship=third_party`; the
  harness is unmistakably `lm-evaluation-harness` (from the `acc,none` /
  `acc_stderr,none` / `bootstrap_iters` keys), so `eval_library` names it.
- **evaluation_id** is keyed on the result file's timestamp (parsed from the
  filename), so re-ingesting the same run is idempotent. Root-level 2-segment
  baselines (hand-curated closed-model paper numbers, e.g. `GPT-4/results_*.json`)
  have different provenance and are skipped; hidden dirs and Jupyter
  `*-checkpoint.json` files are filtered out.

## Run

```bash
uv run python -m every_eval_ever.adapters.open_medical_llm.adapter --output-dir /tmp/eee-omll [--limit N]
uv run python -m every_eval_ever validate /tmp/eee-omll
```

Options: `--output-dir` (default `data/open-medical-llm`), `--limit N` (first N
models), `--workers` (concurrent fetches, default 8).
