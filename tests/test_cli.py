import json

from test_model import tiny_model


def test_cli_inspect_run_and_saved_artifacts(tmp_path, capsys):
    from semantics_operator.cli import main

    model_dir = tmp_path / "weights"
    lm = tiny_model()
    lm.model.save_pretrained(model_dir)
    lm.tokenizer.save_pretrained(model_dir)
    env = tmp_path / ".env"
    env.write_text(f'SO_MODELS_DIR="{tmp_path.as_posix()}"\nSO_MODEL=test\n')
    config = tmp_path / "experiment.toml"
    config.write_text(
        '[models.test]\npath="weights"\ndevice="cpu"\n'
        '[experiment]\nsteps=1\nrank=2\noutput_dir="results"\n'
    )
    args = ["--config", str(config), "--env-file", str(env)]
    assert main(["inspect", *args]) == 0
    assert "down_proj" in capsys.readouterr().out
    assert main(["run", *args]) == 0
    reports = list((tmp_path / "results").glob("*/report.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text())
    from safetensors.torch import load_file

    patches = load_file(reports[0].parent / "operators.safetensors")
    assert set(patches) == {
        f"{key}.{factor}" for key in ("relay_0", "relay_1", "lamp_1") for factor in ("a", "b")
    }
    assert report["rollback"]["max_score_difference"] == 0
    assert (reports[0].parent / "summary.md").is_file()
    assert (reports[0].parent / "representations.safetensors").is_file()


def test_cli_missing_model_gives_actionable_error(tmp_path, capsys):
    from semantics_operator.cli import main

    config = tmp_path / "config.toml"
    config.write_text('default_model="missing"\n[models.missing]\npath="absent"\n')
    assert main(["inspect", "--config", str(config), "--env-file", str(tmp_path / ".env")]) == 2
    assert "config.json" in capsys.readouterr().err


def test_cli_steer_saves_vectors(tmp_path):
    from semantics_operator.cli import main

    lm = tiny_model()
    lm.model.save_pretrained(tmp_path / "weights")
    lm.tokenizer.save_pretrained(tmp_path / "weights")
    config = tmp_path / "config.toml"
    config.write_text(
        '[models.test]\npath="weights"\ndevice="cpu"\n[experiment]\noutput_dir="runs"\n'
    )
    env = tmp_path / ".env"
    env.write_text(f"SO_MODELS_DIR={tmp_path.as_posix()}\nSO_MODEL=test\n")
    assert (
        main(
            [
                "steer",
                "--config",
                str(config),
                "--env-file",
                str(env),
                "--layers",
                "model.layers.0.mlp.down_proj",
                "--strengths",
                "0",
            ]
        )
        == 0
    )
    report_path = next((tmp_path / "runs").glob("*/report.json"))
    report = json.loads(report_path.read_text())
    assert report["experiment"] == "contrastive_activation_v1"
    assert all(choice["alpha"] == 0 for choice in report["selected"].values())
    assert (report_path.parent / "vectors.safetensors").is_file()
