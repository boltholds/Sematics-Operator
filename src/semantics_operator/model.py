"""Offline Hugging Face causal LM adapter with differentiable candidate scoring."""

from pathlib import Path

import torch
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import Settings


class LocalLanguageModel:
    def __init__(self, model, tokenizer, max_length: int):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.max_length = max_length
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer needs a pad or eos token for batched scoring")
        self.pad_id = (
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        )
        self.device = next(model.parameters()).device

    @classmethod
    def load(cls, settings: Settings):
        path = Path(settings.model_path)
        is_gguf = path.suffix.lower() == ".gguf"
        if not is_gguf and not (path / "config.json").is_file():
            raise ValueError(f"Local HF model directory missing config.json: {path}")
        device = settings.device
        if device == "auto":
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        if device == "cpu" and settings.dtype == "float16":
            raise ValueError("Use float32 or bfloat16 for CPU experiments")
        if is_gguf:
            from .gguf_loader import load_gguf

            model, tokenizer = load_gguf(path, getattr(torch, settings.dtype))
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                path, local_files_only=True, trust_remote_code=False
            )
            model = AutoModelForCausalLM.from_pretrained(
                path,
                local_files_only=True,
                trust_remote_code=False,
                use_safetensors=True,
                dtype=getattr(torch, settings.dtype),
                attn_implementation="eager",
            )
        if getattr(model.config, "quantization_config", None):
            raise ValueError("v0.1 requires unquantized floating-point HF weights")
        model.to(device).eval()
        return cls(model, tokenizer, settings.max_length)

    def linear_modules(self) -> dict[str, tuple[int, int]]:
        return {
            name: tuple(module.weight.shape)
            for name, module in self.model.named_modules()
            if type(module) is nn.Linear
        }

    def choose_target(self, requested: str) -> str:
        modules = self.linear_modules()
        if requested:
            if requested not in modules:
                raise ValueError(f"Not an editable Linear module: {requested}. Run semop inspect.")
            if requested == "lm_head":
                raise ValueError("Output-head editing is excluded; select an internal layer")
            return requested
        candidates = [
            name
            for name in modules
            if name.endswith(("down_proj", "fc2", "dense_4h_to_h", "feed_forward.w2"))
        ]
        if not candidates:
            raise ValueError(
                "No recognized MLP projection; run semop inspect and set target_module"
            )
        return candidates[len(candidates) // 2]

    def _prompt_ids(self, prompt: str) -> list[int]:
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True
            )
        ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        if not ids:
            raise ValueError("Prompt encoded to an empty sequence")
        return ids

    def _batch(self, sequences: list[list[int]]) -> tuple[Tensor, Tensor]:
        length = max(map(len, sequences))
        if length > self.max_length:
            raise ValueError(
                f"Input length {length} exceeds max_length={self.max_length}; "
                "increase it explicitly. Inputs are never silently truncated."
            )
        ids = torch.full(
            (len(sequences), length), self.pad_id, dtype=torch.long, device=self.device
        )
        mask = torch.zeros_like(ids)
        for row, seq in enumerate(sequences):
            ids[row, : len(seq)] = torch.tensor(seq, device=self.device)
            mask[row, : len(seq)] = 1
        return ids, mask

    def scores(self, prompts: list[str], *, candidates=(" 0", " 1")) -> Tensor:
        """Sum token log-probabilities of continuations ' 0' and ' 1', not sampling.

        Prompt and continuations are tokenized separately to make the conditional
        boundary explicit, including tokenizers whose BPE merges span that boundary.
        All continuation tokens are scored; no single-token digit assumption.
        """
        if not prompts:
            raise ValueError("At least one prompt is required")
        if len(candidates) != 2 or any(not isinstance(c, str) or not c for c in candidates):
            raise ValueError("Exactly two nonempty candidate strings are required")
        sequences, rows, positions, targets, owners = [], [], [], [], []
        for prompt in prompts:
            prefix = self._prompt_ids(prompt)
            for candidate in candidates:
                suffix = self.tokenizer.encode(candidate, add_special_tokens=False)
                if not suffix:
                    raise ValueError("Empty candidate tokenization")
                row = len(sequences)
                sequences.append(prefix + suffix)
                for j, token in enumerate(suffix):
                    rows.append(row)
                    positions.append(len(prefix) + j - 1)
                    targets.append(token)
                    owners.append(row)
        ids, mask = self._batch(sequences)
        logits = self.model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        selected = logits[rows, positions].float().log_softmax(-1)
        target_ids = torch.tensor(targets, device=self.device)
        logp = selected.gather(1, target_ids[:, None]).squeeze(1)
        scores = torch.zeros(len(sequences), device=self.device)
        scores = scores.index_add(0, torch.tensor(owners, device=self.device), logp)
        return scores.reshape(len(prompts), 2)

    @torch.no_grad()
    def generate_greedy(self, prompts: list[str], *, max_new_tokens=16) -> list[dict]:
        """Unconstrained argmax decoding, one prompt at a time, without a forced space.

        Uses the same forward/chat formatting as scoring and no KV cache, including
        hybrid GGUF models. This intentionally does not apply HF logits processors.
        """
        if not prompts or type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("Provide prompts and a positive max_new_tokens")
        prefixes = [self._prompt_ids(p) for p in prompts]
        if any(len(p) + max_new_tokens > self.max_length for p in prefixes):
            raise ValueError(
                "Prompt plus max_new_tokens exceeds max_length; increase it explicitly"
            )
        eos = None
        for config in (
            getattr(self.model, "generation_config", None),
            self.model.config,
            self.tokenizer,
        ):
            eos = getattr(config, "eos_token_id", None)
            if eos is not None:
                break
        stop_ids = set(eos if isinstance(eos, (list, tuple)) else [] if eos is None else [eos])
        results = []
        for prefix in prefixes:
            generated, reason = [], "max_new_tokens"
            for _ in range(max_new_tokens):
                ids, mask = self._batch([prefix + generated])
                logits = self.model(input_ids=ids, attention_mask=mask, use_cache=False).logits
                if not torch.isfinite(logits[0, -1]).all():
                    raise FloatingPointError("Non-finite generation logits")
                token = int(logits[0, -1].argmax())
                generated.append(token)
                if token in stop_ids:
                    reason = "eos"
                    break
            results.append(
                {
                    "text": self.tokenizer.decode(generated, skip_special_tokens=True),
                    "token_ids": generated,
                    "stop_reason": reason,
                }
            )
        return results

    @torch.no_grad()
    def representations(self, prompts: list[str], target: str) -> Tensor:
        """Capture the selected module output at each last non-padding prompt token."""
        captures: list[Tensor] = []
        handle = self.model.get_submodule(target).register_forward_hook(
            lambda module, args, output: captures.append(output.detach())
        )
        try:
            sequences = [self._prompt_ids(p) for p in prompts]
            ids, mask = self._batch(sequences)
            self.model(input_ids=ids, attention_mask=mask, use_cache=False)
            if len(captures) != 1:
                raise ValueError("Target must execute exactly once per forward")
            tensor = captures[0]
            return (
                tensor[torch.arange(len(sequences), device=self.device), mask.sum(-1) - 1]
                .float()
                .cpu()
            )
        finally:
            handle.remove()
