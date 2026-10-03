"""Real, small GGUF files generated locally; no model downloads or mocked loaders."""

from pathlib import Path

import gguf
import numpy as np
import torch
from tokenizers.pre_tokenizers import ByteLevel
from transformers import Lfm2Config, Lfm2ForCausalLM


def write_lfm2_gguf(path: Path, *, quantized: bool = True, omit: str = ""):
    torch.manual_seed(17)
    tokens = ["<pad>", "<bos>", "<eos>", "<unk>", "<start>", "<end>"]
    tokens += sorted(ByteLevel.alphabet())
    tokens += ["ab"]
    tokens += [f"<unused_{i}>" for i in range(288 - len(tokens))]
    config = Lfm2Config(
        vocab_size=len(tokens),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        full_attn_idxs=[1],
        block_auto_adjust_ff_dim=False,
        conv_L_cache=3,
        norm_eps=0.003,
        max_position_embeddings=1024,
        rope_theta=10000.0,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = Lfm2ForCausalLM(config).eval()
    writer = gguf.GGUFWriter(path, "lfm2")
    writer.add_name("tiny-lfm2-fixture")
    writer.add_context_length(1024)
    writer.add_embedding_length(32)
    writer.add_feed_forward_length(64)
    writer.add_block_count(2)
    writer.add_head_count(4)
    writer.add_head_count_kv([0, 2])
    writer.add_layer_norm_rms_eps(0.003)
    writer.add_rope_freq_base(10000.0)
    writer.add_uint32("lfm2.shortconv.l_cache", 3)
    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("lfm2")
    writer.add_token_list(tokens)
    writer.add_token_types([3 if t.startswith("<") else 1 for t in tokens])
    writer.add_token_merges(["a b"])
    writer.add_bos_token_id(1)
    writer.add_eos_token_id(2)
    writer.add_pad_token_id(0)
    writer.add_unk_token_id(3)
    writer.add_add_bos_token(True)
    writer.add_add_eos_token(False)
    writer.add_chat_template(
        "{{ bos_token }}{% for m in messages %}<start>{{ m['role'] }}\n"
        "{{ m['content'] }}<end>\n{% endfor %}"
        "{% if add_generation_prompt %}<start>assistant\n{% endif %}"
    )
    names = gguf.get_tensor_name_map(gguf.MODEL_ARCH.LFM2, 2)
    expected = {}
    for name, tensor in model.state_dict().items():
        if name == "lm_head.weight":
            continue  # Tied embedding, absent output.weight is intentional.
        stem, suffix = name.rsplit(".", 1)
        gguf_name = names.get_name(stem)
        assert gguf_name is not None, name
        gguf_name += "." + suffix
        if gguf_name == omit:
            continue
        array = tensor.detach().float().numpy()
        if "shortconv.conv.weight" in gguf_name:
            array = array.squeeze(1)
        if quantized and array.ndim == 2 and array.shape[-1] % 32 == 0:
            data = gguf.quants.quantize(array, gguf.GGMLQuantizationType.Q8_0)
            writer.add_tensor(gguf_name, data, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
            restored = gguf.quants.dequantize(data, gguf.GGMLQuantizationType.Q8_0)
        else:
            writer.add_tensor(gguf_name, np.ascontiguousarray(array))
            restored = array
        expected[name] = torch.from_numpy(restored.copy()).reshape(tensor.shape)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    expected["lm_head.weight"] = expected["model.embed_tokens.weight"]
    model.load_state_dict(expected, strict=not bool(omit))
    return model, expected
