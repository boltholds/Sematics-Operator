import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class ConfigTests(unittest.TestCase):
    def test_profile_path_is_relative_to_models_dir_and_env_file(self):
        from semantics_operator.config import load_settings

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / ".env").write_text("SO_MODELS_DIR=./weights\nSO_MODEL=small\n")
            (root / "config.toml").write_text('[models.small]\npath="model-a"\n')
            with patch.dict(os.environ, {}, clear=True):
                cfg = load_settings(root / "config.toml", root / ".env")
            self.assertEqual(cfg.model_path, root / "weights" / "model-a")

    def test_shell_env_wins_and_missing_profile_fails(self):
        from semantics_operator.config import load_settings

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / ".env").write_text("SO_MODEL=bad\n")
            (root / "config.toml").write_text('[models.good]\npath="weights"\n')
            with patch.dict(os.environ, {"SO_MODEL": "good"}, clear=True):
                self.assertEqual(load_settings(root / "config.toml", root / ".env").profile, "good")
            with (
                patch.dict(os.environ, {}, clear=True),
                self.assertRaisesRegex(ValueError, "Unknown model"),
            ):
                load_settings(root / "config.toml", root / ".env")

    def test_unknown_profile_key_is_rejected_instead_of_changing_target(self):
        from semantics_operator.config import load_settings

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "config.toml").write_text(
                'default_model="small"\n[models.small]\npath="weights"\ntarget_modul="wrong"\n'
            )
            with (
                patch.dict(os.environ, {}, clear=True),
                self.assertRaisesRegex(ValueError, "Unknown model settings"),
            ):
                load_settings(root / "config.toml", root / ".env")
