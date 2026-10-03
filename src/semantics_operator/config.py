"""TOML profiles and .env paths; process environment takes precedence."""

import math
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values


@dataclass(frozen=True)
class Settings:
    profile: str
    model_path: Path
    target_module: str = ""
    device: str = "auto"
    dtype: str = "float32"
    rank: int = 4
    steps: int = 30
    learning_rate: float = 0.003
    locality_weight: float = 0.5
    regularization: float = 0.0001
    max_length: int = 512
    seed: int = 42
    output_dir: Path = Path("runs")

    def __post_init__(self):
        if self.device not in ("auto", "cpu", "cuda", "mps"):
            raise ValueError("device must be auto, cpu, cuda or mps")
        if self.dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError("Unsupported dtype")
        if any(type(v) is not int or v < 1 for v in (self.rank, self.steps, self.max_length)):
            raise ValueError("rank, steps and max_length must be positive integers")
        for name in ("learning_rate", "locality_weight", "regularization"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0 or (name == "learning_rate" and value == 0):
                raise ValueError(f"Invalid {name}")


def load_settings(config_path: Path, env_path: Path, profile: str = "") -> Settings:
    config_path, env_path = config_path.resolve(), env_path.resolve()
    with config_path.open("rb") as stream:
        document = tomllib.load(stream)
    env = {**dotenv_values(env_path), **os.environ}
    selected = profile or env.get("SO_MODEL") or document.get("default_model", "")
    models = document.get("models", {})
    if selected not in models:
        raise ValueError(f"Unknown model profile {selected!r}; available: {', '.join(models)}")
    model = models[selected]
    unknown_model = set(model) - {"path", "target_module", "device", "dtype"}
    if unknown_model:
        raise ValueError(f"Unknown model settings: {sorted(unknown_model)}")
    if not isinstance(model.get("path"), str) or not model["path"].strip():
        raise ValueError("Model profile requires a nonempty path")
    root = Path(env.get("SO_MODELS_DIR") or ".").expanduser()
    if not root.is_absolute():
        root = env_path.parent / root
    path = Path(model["path"]).expanduser()
    if not path.is_absolute():
        path = root / path
    experiment = document.get("experiment", {})
    allowed = {
        "rank",
        "steps",
        "learning_rate",
        "locality_weight",
        "regularization",
        "max_length",
        "seed",
        "output_dir",
    }
    unknown = set(experiment) - allowed
    if unknown:
        raise ValueError(f"Unknown experiment settings: {sorted(unknown)}")
    output = Path(experiment.get("output_dir", "runs")).expanduser()
    if not output.is_absolute():
        output = config_path.parent / output
    options = {k: v for k, v in experiment.items() if k != "output_dir"}
    return Settings(
        profile=selected,
        model_path=path.resolve(),
        target_module=model.get("target_module", ""),
        device=env.get("SO_DEVICE") or model.get("device", "auto"),
        dtype=env.get("SO_DTYPE") or model.get("dtype", "float32"),
        output_dir=output.resolve(),
        **options,
    )
