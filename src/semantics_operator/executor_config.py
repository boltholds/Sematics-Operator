import math
import os
import tomllib
from dataclasses import dataclass, fields, replace
from pathlib import Path

from dotenv import dotenv_values


@dataclass(frozen=True)
class ExecutorConfig:
    steps: int = 300
    width: int = 64
    heads: int = 4
    blocks: int = 2
    batch_size: int = 32
    train_graphs: int = 64
    eval_graphs: int = 16
    seeds: tuple[int, ...] = (42, 43, 44)
    dataset_seed: int = 20261004
    learning_rate: float = 0.003
    preservation_weight: float = 1.0
    validation_every: int = 25
    threads: int = 2
    device: str = "cpu"
    output_dir: Path = Path("runs")

    def validate(self):
        for key in (
            "steps",
            "width",
            "heads",
            "blocks",
            "batch_size",
            "train_graphs",
            "eval_graphs",
            "validation_every",
            "threads",
        ):
            value = getattr(self, key)
            if type(value) is not int or value < 1:
                raise ValueError(f"executor.{key} must be a positive integer")
        if self.width % self.heads:
            raise ValueError("executor.width must be divisible by heads")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("executor.seeds must be nonempty and unique")
        if any(type(s) is not int or s < 0 for s in (*self.seeds, self.dataset_seed)):
            raise ValueError("Seeds must be nonnegative integers")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.preservation_weight) or self.preservation_weight < 0:
            raise ValueError("preservation_weight must be finite and nonnegative")
        if self.device not in ("auto", "cpu", "cuda", "mps"):
            raise ValueError("executor.device must be auto, cpu, cuda or mps")
        return self


def load_executor_config(path, env_file, **overrides):
    path = Path(path)
    with path.open("rb") as stream:
        data = tomllib.load(stream).get("executor", {})
    unknown = set(data) - {f.name for f in fields(ExecutorConfig)}
    if unknown:
        raise ValueError(f"Unknown executor config fields: {sorted(unknown)}")
    data["seeds"] = tuple(data.get("seeds", ExecutorConfig.seeds))
    data["output_dir"] = (path.parent / data.get("output_dir", "../runs")).resolve()
    env = {**dotenv_values(env_file), **os.environ}
    if env.get("SO_DEVICE"):
        data["device"] = env["SO_DEVICE"]
    cfg = ExecutorConfig(**data)
    return replace(cfg, **{k: v for k, v in overrides.items() if v is not None}).validate()
