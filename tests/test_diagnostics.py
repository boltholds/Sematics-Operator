import json

from test_model import tiny_model


def test_diagnostic_metrics_keep_invalid_generations_and_zero_one_errors():
    from semantics_operator.diagnostics import answer_metrics

    result = answer_metrics(
        [
            {"expected": 0, "prediction": 0, "node": "relay", "style": "default"},
            {"expected": 0, "prediction": 1, "node": "relay", "style": "default"},
            {"expected": 1, "prediction": None, "node": "lamp", "style": "verbal"},
            {"expected": 1, "prediction": 1, "node": "lamp", "style": "verbal"},
        ]
    )
    assert result["overall"] == {
        "count": 4,
        "accuracy": 0.5,
        "errors": 2,
        "invalid": 1,
        "incomplete": 0,
        "format_errors": 0,
    }
    assert result["by_label"]["0"]["errors"] == 1
    assert result["by_label"]["1"]["invalid"] == 1
    assert result["prediction_counts"] == {"0": 1, "1": 2, "invalid": 1}


def test_diagnose_cli_saves_raw_generation_and_both_candidate_formats(tmp_path):
    from semantics_operator.cli import main

    lm = tiny_model()
    lm.model.save_pretrained(tmp_path / "weights")
    lm.tokenizer.save_pretrained(tmp_path / "weights")
    config = tmp_path / "config.toml"
    config.write_text(
        '[models.tiny]\npath="weights"\ndevice="cpu"\n[experiment]\noutput_dir="runs"\n'
    )
    env = tmp_path / ".env"
    env.write_text(f"SO_MODELS_DIR={tmp_path.as_posix()}\nSO_MODEL=tiny\n")
    assert (
        main(
            [
                "diagnose",
                "--config",
                str(config),
                "--env-file",
                str(env),
                "--max-new-tokens",
                "1",
            ]
        )
        == 0
    )
    path = next((tmp_path / "runs").glob("*-diagnose-*/report.json"))
    report = json.loads(path.read_text())
    assert report["experiment"] == "base_answer_diagnostics_v2"
    assert report["max_new_tokens"] == 1
    for schemes in report["splits"].values():
        for modes in schemes.values():
            assert set(modes) == {"spaced_candidates", "bare_candidates", "greedy", "agreement"}
            assert set(modes["greedy"]["by_style"]) == {"default", "verbal", "query_first"}
            for record in modes["greedy"]["records"]:
                assert "text" in record and "token_ids" in record and "prompt" in record
                assert record["stop_reason"] in ("eos", "max_new_tokens")
    assert set(report["splits"]["train"]) == {"and_copy"}
    assert set(report["splits"]["validation"]) == {"and_copy"}
    assert len(report["splits"]["test"]) == 8
    assert (path.parent / "summary.md").is_file()
