"""Offline GGUF -> floating-point Transformers loading for differentiable edits.

GGUF is an input checkpoint format here, not a quantized training runtime. The
original file is read-only. LFM2 tokenization follows llama.cpp's lfm2/llama3 BPE
pre-tokenizer; the vocabulary, merges, special IDs and chat template come from GGUF.
"""

import warnings
from pathlib import Path

from gguf import GGUFReader
from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, pre_tokenizers, processors
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast
from transformers.modeling_gguf_pytorch_utils import (
    GGUF_SUPPORTED_ARCHITECTURES,
    load_gguf_checkpoint,
)

LFM2_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|"
    r" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


def field(reader: GGUFReader, name: str, default=None):
    entry = reader.get_field(name)
    return default if entry is None else entry.contents()


def lfm2_tokenizer(reader: GGUFReader) -> PreTrainedTokenizerFast:
    if field(reader, "tokenizer.ggml.model") != "gpt2":
        raise ValueError("LFM2 GGUF requires its gpt2/BPE tokenizer metadata")
    pre = field(reader, "tokenizer.ggml.pre")
    if pre not in ("lfm2", "llama3", "llama-bpe", "llama-v3"):
        raise ValueError(f"Unsupported LFM2 GGUF pre-tokenizer: {pre!r}")
    tokens = field(reader, "tokenizer.ggml.tokens", [])
    merges = field(reader, "tokenizer.ggml.merges", [])
    types = field(reader, "tokenizer.ggml.token_type", [])
    if not tokens or len(set(tokens)) != len(tokens) or len(types) != len(tokens):
        raise ValueError("GGUF tokenizer needs unique tokens and matching token_type metadata")
    pairs = [tuple(merge.split(" ")) for merge in merges]
    if any(len(pair) != 2 for pair in pairs):
        raise ValueError("Malformed GGUF BPE merges")
    backend = Tokenizer(
        models.BPE(
            vocab={token: i for i, token in enumerate(tokens)},
            merges=pairs,
            byte_fallback=False,
            ignore_merges=True,
        )
    )
    backend.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(LFM2_PATTERN), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    backend.decoder = decoders.ByteLevel()
    special = [
        AddedToken(token, normalized=False, special=True)
        for token, kind in zip(tokens, types)
        if kind == 3
    ]
    backend.add_special_tokens(special)
    backend.add_tokens(
        [
            AddedToken(token, normalized=False, special=False)
            for token, kind in zip(tokens, types)
            if kind == 4
        ]
    )
    kwargs = {"clean_up_tokenization_spaces": False}
    for role, key in [("bos", "bos"), ("eos", "eos"), ("pad", "padding"), ("unk", "unknown")]:
        index = field(reader, f"tokenizer.ggml.{key}_token_id")
        if index is not None:
            if not isinstance(index, int) or not 0 <= index < len(tokens):
                raise ValueError(f"Invalid GGUF {role} token ID: {index}")
            kwargs[f"{role}_token"] = tokens[index]
    prefix, suffix, special_pairs = [], [], []
    for role, enabled, location in [
        ("bos", field(reader, "tokenizer.ggml.add_bos_token", True), prefix),
        ("eos", field(reader, "tokenizer.ggml.add_eos_token", False), suffix),
    ]:
        if enabled:
            token = kwargs.get(f"{role}_token")
            if token is None:
                raise ValueError(f"GGUF requests {role} insertion but provides no token ID")
            location.append(token)
            special_pairs.append((token, tokens.index(token)))
    backend.post_processor = processors.TemplateProcessing(
        single=prefix + ["$A"] + suffix,
        pair=prefix + ["$A"] + suffix + prefix + ["$B:1"] + suffix,
        special_tokens=special_pairs,
    )
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, **kwargs)
    template = field(reader, "tokenizer.chat_template")
    if template is not None:
        tokenizer.chat_template = template
    return tokenizer


def load_gguf(path: Path, dtype):
    if not path.is_file():
        raise ValueError(f"Local GGUF file not found: {path}")
    reader = GGUFReader(str(path), mode="r")
    architecture = field(reader, "general.architecture")
    if architecture not in GGUF_SUPPORTED_ARCHITECTURES:
        raise ValueError(
            f"Unsupported GGUF architecture {architecture!r} in installed Transformers"
        )
    if field(reader, "split.count", 1) != 1:
        raise ValueError("Split GGUF checkpoints are not supported; select a single-file GGUF")
    metadata = load_gguf_checkpoint(str(path), return_tensors=False)
    config_data = metadata["config"]
    if architecture == "lfm2":
        # Transformers 4.57 maps this GGUF key to the Llama spelling, which LFM2 ignores.
        config_data["norm_eps"] = config_data.pop("rms_norm_eps", 1e-5)
        config_data["tie_word_embeddings"] = not any(
            t.name == "output.weight" for t in reader.tensors
        )
        tokenizer = lfm2_tokenizer(reader)
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                path.parent, gguf_file=path.name, local_files_only=True, trust_remote_code=False
            )
        except (KeyError, NotImplementedError) as error:
            raise ValueError(
                f"GGUF tokenizer for architecture {architecture!r} is unsupported"
            ) from error
    for role in ("bos", "eos", "pad"):
        token_id = getattr(tokenizer, f"{role}_token_id")
        if token_id is not None:
            config_data[f"{role}_token_id"] = token_id
    config = AutoConfig.for_model(**config_data)
    parameter_count = sum(int(t.n_elements) for t in reader.tensors)
    del reader
    warnings.warn(
        f"GGUF will be dequantized for gradient-based weight edits. "
        f"FP32 tensors alone require about {parameter_count * 4 / 2**30:.2f} GiB of RAM "
        "(loading and activations need additional memory). The source file is not modified.",
        UserWarning,
        stacklevel=2,
    )
    model, info = AutoModelForCausalLM.from_pretrained(
        path.parent,
        gguf_file=path.name,
        config=config,
        local_files_only=True,
        trust_remote_code=False,
        dtype=dtype,
        attn_implementation="eager",
        output_loading_info=True,
    )
    problems = {key: value for key, value in info.items() if value}
    if problems:
        raise ValueError(f"incomplete GGUF weights: {problems}")
    return model, tokenizer
