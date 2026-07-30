"""Tests for the WILD-raw adapter (utils/wild/adapter.py). No network — builds a
tiny local parquet and runs the adapter over it."""
import argparse
import hashlib
import json
import math

import pytest

pytest.importorskip(
    'pyarrow',
    reason='pyarrow not installed; the wild adapter needs it (uv sync --extra wild)',
)

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from every_eval_ever.validate import validate_file  # noqa: E402
from utils.wild import adapter  # noqa: E402


def _synth_parquet(path):
    rows = []
    for model in ["openai/gpt-x", "01-ai/Yi-1.5-34B-Chat"]:
        for subtask in ["algebra", "logic"]:
            for i in range(3):
                score = 1 if i % 2 == 0 else 0
                # the assistant turn is the model's FULL generation (chain-of-thought);
                # it deliberately DIFFERS from the extracted `answer` ("4"/"5") and the
                # scorer's parsed answer so tests can prove output.raw != extracted_value.
                gen = f"Step by step: two plus two is four. Final answer: {'4' if score else '5'}."
                convo = json.dumps([{"role": "user", "content": "What is 2+2?"},
                                    {"role": "assistant", "content": gen}])
                rows.append(dict(
                    model=model, task="mmlu", subtask=subtask,
                    item_id=f"{model[:3]}{subtask[:2]}{i}", score=score,
                    input_tokens=100 + i, output_tokens=20 + i, conversation=convo,
                    stop_reason="stop", target="4", answer="4" if score else "5",
                    scores=json.dumps({"match": {"value": "C" if score else "I",
                                                 "answer": "the answer is 4"}})))
    pq.write_table(pa.Table.from_pylist(rows), str(path))


def _args(parquet, out, **kw):
    base = dict(parquet=[str(parquet)], output_dir=out, limit_shards=None,
                models=None, include_instances=False, max_instances=None,
                retrieved_timestamp="1700000000.0", evaluation_timestamp="1780000000.0",
                revision=None)
    base.update(kw)
    return argparse.Namespace(**base)


def test_aggregates(tmp_path):
    pqt = tmp_path / "w.parquet"
    _synth_parquet(pqt)
    out = tmp_path / "out"
    n = adapter.run(_args(pqt, out))
    assert n == 2  # 2 models x 1 benchmark
    files = list(out.rglob("*.json"))
    assert len(files) == 2
    for f in files:
        report = validate_file(f)
        assert report.valid, report.errors
    log = json.loads(next((out / "openai" / "gpt-x").glob("*.json")).read_text())
    names = {r["evaluation_name"] for r in log["evaluation_results"]}
    assert names == {"wild.mmlu", "wild.mmlu.algebra", "wild.mmlu.logic"}
    overall = next(r for r in log["evaluation_results"] if r["evaluation_name"] == "wild.mmlu")
    assert overall["metric_config"]["score_type"] == "continuous"
    assert (overall["metric_config"]["min_score"], overall["metric_config"]["max_score"]) == (0.0, 1.0)
    assert abs(overall["score_details"]["score"] - 2 / 3) < 1e-9
    # analytic proportion SE = sqrt(p(1-p)/n), p=2/3 over n=6 items — regression guard
    unc = overall["score_details"]["uncertainty"]
    assert abs(unc["standard_error"]["value"] - math.sqrt((2 / 3) * (1 / 3) / 6)) < 1e-9
    assert unc["num_samples"] == 6
    assert log["source_metadata"]["source_type"] == "evaluation_run"
    assert log["model_info"]["id"] == "openai/gpt-x"
    assert log["evaluation_id"] == "wild/openai_gpt-x/mmlu/1780000000.0"  # keyed on eval time
    assert log["retrieved_timestamp"] == "1700000000.0"       # record-creation time
    assert log["evaluation_timestamp"] == "1780000000.0"      # when the eval ran
    assert log["eval_library"]["name"] == "inspect_ai"
    # source_data points at the benchmark's dataset repo, not WILD-raw
    assert overall["source_data"]["source_type"] == "hf_dataset"
    assert overall["source_data"]["hf_repo"] == "cais/mmlu"


def test_single_subtask_dedup(tmp_path):
    # a task whose only subtask is "general" must emit ONLY wild.<task> (no dup leaf),
    # and instances must attach to the overall result id (task), not task::general.
    convo = json.dumps([{"role": "user", "content": "Q?"},
                        {"role": "assistant", "content": "ANSWER: C"}])
    rows = [dict(model="openai/gpt-x", task="arc_challenge", subtask="general",
                 item_id=f"i{i}", score=i % 2, input_tokens=10, output_tokens=2,
                 conversation=convo, stop_reason="stop", target="C", answer="C",
                 scores=json.dumps({"choice": {"value": "C", "answer": "ANSWER: C"}}))
            for i in range(4)]
    pqt = tmp_path / "w.parquet"
    pq.write_table(pa.Table.from_pylist(rows), str(pqt))
    out = tmp_path / "out"
    adapter.run(_args(pqt, out, include_instances=True))
    log = json.loads(next(out.rglob("*.json")).read_text())
    names = [r["evaluation_name"] for r in log["evaluation_results"]]
    assert names == ["wild.arc_challenge"]  # deduped: no wild.arc_challenge.general
    inst = json.loads(next(out.rglob("*_samples.jsonl")).read_text().splitlines()[0])
    assert inst["evaluation_result_id"] == "arc_challenge"          # FK resolves to overall
    assert inst["input"]["raw"] == "Q?"                              # answer NOT leaked in
    assert inst["output"]["raw"] == ["ANSWER: C"]                    # full generation = assistant turn
    assert inst["answer_attribution"][0]["extraction_method"] == "choice"  # real scorer
    assert "sample_hash" in inst
    # source_data for arc_challenge -> the AI2 ARC dataset
    assert log["evaluation_results"][0]["source_data"]["hf_repo"] == "allenai/ai2_arc"


def test_instances(tmp_path):
    pqt = tmp_path / "w.parquet"
    _synth_parquet(pqt)
    out = tmp_path / "out"
    adapter.run(_args(pqt, out, include_instances=True))
    samples = list(out.rglob("*_samples.jsonl"))
    assert len(samples) == 2
    for s in samples:
        report = validate_file(s)
        assert report.valid, report.errors
    # aggregate points at its sidecar
    agg = next((out / "openai" / "gpt-x").glob("*.json"))
    log = json.loads(agg.read_text())
    det = log["detailed_evaluation_results"]
    assert det["format"] == "jsonl" and det["total_rows"] == 6
    inst = json.loads(next((out / "openai" / "gpt-x").glob("*_samples.jsonl")).read_text().splitlines()[0])
    assert inst["interaction_type"] == "single_turn"
    assert inst["evaluation"]["is_correct"] in (True, False)
    assert inst["token_usage"]["total_tokens"] == inst["token_usage"]["input_tokens"] + inst["token_usage"]["output_tokens"]
    assert inst["evaluation_name"].startswith("wild.mmlu.")
    # output.raw is the model's FULL generation (assistant turn), NOT the parsed answer
    full = inst["output"]["raw"]
    assert len(full) == 1 and full[0].startswith("Step by step")
    ev = inst["answer_attribution"][0]["extracted_value"]
    assert ev in ("4", "5")
    assert ev != full[0]                              # generation != extracted answer (regression guard)
    # sample_hash uses the canonical cross-adapter recipe over (input.raw, reference)
    assert inst["sample_hash"] == adapter._sample_hash(inst["input"]["raw"], inst["input"]["reference"])


def test_sample_hash_is_canonical():
    # locks the recipe to the skill's templates/instance_sidecar._sample_hash
    expected = hashlib.sha256(
        json.dumps({"raw": "Q?", "reference": ["C"]}, sort_keys=True,
                   separators=(",", ":")).encode("utf-8")).hexdigest()
    assert adapter._sample_hash("Q?", ["C"]) == expected


def test_split_conversation_separates_prompt_and_generation():
    convo = json.dumps([{"role": "system", "content": "sys"},
                        {"role": "user", "content": "Q?"},
                        {"role": "assistant", "content": "the full model answer"}])
    prompt, generation = adapter._split_conversation(convo)
    assert prompt == "sys\n\nQ?"                       # user + system only, no assistant
    assert generation == ["the full model answer"]     # assistant turn -> output.raw


def test_local_run_provenance_no_false_revision(tmp_path):
    # a local --parquet run must NOT stamp dataset_revision='main' (false remote provenance)
    pqt = tmp_path / "w.parquet"
    _synth_parquet(pqt)
    out = tmp_path / "out"
    adapter.run(_args(pqt, out))
    log = json.loads(next((out / "openai" / "gpt-x").glob("*.json")).read_text())
    ad = log["source_metadata"]["additional_details"]
    assert "dataset_revision" not in ad                # unknown for a local file
    assert "local" in ad.get("dataset_source", "").lower()
