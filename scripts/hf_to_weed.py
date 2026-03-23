#!/usr/bin/env python3
"""
hf_to_weed.py — Convert HuggingFace safetensors models to Weed binary format.

(C) Daniel Strano and the Qrack contributors 2026.

This file was produced almost in its entirety, verbatim, by (Anthropic) Claude.

Use of this source code is governed by an MIT-style license that can be
found in the LICENSE file or at https://opensource.org/licenses/MIT.

Usage:
    python3 hf_to_weed.py --model_dir <path_to_hf_model> --output <output.weed>
    python3 hf_to_weed.py --model_dir <path_to_hf_model> --output <output.weed> --arch gpt2
    python3 hf_to_weed.py --list_keys --model_dir <path_to_hf_model>

Supported architectures: gpt2, bert, generic_transformer
"""

import argparse
import json
import struct
import sys
from pathlib import Path

try:
    from safetensors import safe_open
    import numpy as np
except ImportError:
    print("Missing dependencies. Install with:")
    print("    pip install safetensors numpy")
    sys.exit(1)

# ---------------------------------------------------------------------------
# ModuleType enum (mirrors module_type.hpp exactly)
# ---------------------------------------------------------------------------
class ModuleType:
    NONE_MODULE_TYPE            = 0
    SEQUENTIAL_T                = 1
    LINEAR_T                    = 2
    RELU_T                      = 3
    SIGMOID_T                   = 4
    TANH_T                      = 5
    DROPOUT_T                   = 6
    LAYERNORM_T                 = 7
    EMBEDDING_T                 = 8
    GRU_T                       = 9
    LSTM_T                      = 10
    MIGRATE_CPU_T               = 11
    MIGRATE_GPU_T               = 12
    SOFTMAX_T                   = 13
    LOGSOFTMAX_T                = 14
    QRACK_NEURON_T              = 15
    QRACK_NEURON_LAYER_T        = 16
    MULTIHEAD_ATTENTION_T       = 17
    TRANSFORMER_ENCODER_LAYER_T = 18
    GELU_T                      = 19
    MEAN_T                      = 20
    MIN_T                       = 21
    MAX_T                       = 22
    RESHAPE_T                   = 23
    VARIANCE_T                  = 24
    STDDEV_T                    = 25
    POSITIONAL_ENCODING_T       = 26
    MEAN_CENTER_T               = 27
    FLATTEN_T                   = 28
    LEARNED_POSITIONAL_ENCODING_T = 29
    RMS_NORM_T  = 30
    ROPE_T      = 31
    SWIGLU_T    = 32

# StorageType enum — matches storage_type.hpp exactly.
class StorageType:
    NONE_STORAGE_TYPE  = 0
    REAL_CPU_DENSE     = 1
    REAL_GPU_DENSE     = 2
    COMPLEX_CPU_DENSE  = 3
    COMPLEX_GPU_DENSE  = 4
    INT_CPU_DENSE      = 5
    INT_GPU_DENSE      = 6
    REAL_CPU_SPARSE    = 7
    COMPLEX_CPU_SPARSE = 8

# ---------------------------------------------------------------------------
# Low-level binary writers
# (All integers are little-endian to match x86 native; adjust if needed.)
# tcapint = uint64_t, symint = int64_t, real1 = float32, bool = uint8
# ---------------------------------------------------------------------------
def w_uint32(f, x):  f.write(struct.pack('<I', int(x)))
def w_uint64(f, x):  f.write(struct.pack('<Q', int(x)))
def w_int64(f, x):   f.write(struct.pack('<q', int(x)))
def w_float32(f, x): f.write(struct.pack('<f', float(x)))
def w_bool(f, x):    f.write(struct.pack('?', bool(x)))

def write_module_type(f, mtype): w_uint32(f, mtype)
def write_tcapint(f, x):         w_uint64(f, x)
def write_symint(f, x):          w_int64(f, x)
def write_real(f, x):            w_float32(f, x)
def write_bool(f, x):            w_bool(f, x)

# ---------------------------------------------------------------------------
# Storage writer
# Matches Storage::save() exactly:
#   write_storage_type(os, stype)      → uint32
#   Serializer::write_tcapint(os, size) → uint64
#   <inheriting class writes elements>
#
# For REAL_CPU_DENSE each element is a float32 (real1).
# We always write CPU dense storage — Weed will migrate to GPU if needed.
# Sparse formats are not generated here (no equivalent in HF dense weights).
# ---------------------------------------------------------------------------
def write_storage(f, arr: np.ndarray):
    # Weed is column-major (Fortran order) — flatten accordingly.
    arr = np.asfortranarray(arr.astype(np.float32))
    flat = arr.flatten(order='F')
    w_uint32(f, StorageType.REAL_CPU_DENSE)  # stype
    write_symint(f, -1)                      # device ID (GPU)
    write_tcapint(f, flat.size)              # size (element count)
    f.write(flat.tobytes())                  # raw float32 elements, column-major

# ---------------------------------------------------------------------------
# Parameter writer
# Matches Parameter::save(): offset | ndim | (shape,stride)* | storage
# Strides are C-contiguous (row-major).
# ---------------------------------------------------------------------------
DEBUG = False  # set to True to print file offsets for each parameter written

def write_parameter(f, arr: np.ndarray, offset: int = 0, label: str = ''):
    # Weed uses column-major (Fortran) strides: stride[0]=1, stride[i]=prod(shape[:i])
    # This matches the stride computation in forward_int exactly.
    arr_f = np.asfortranarray(arr.astype(np.float32))
    shape = list(arr_f.shape)
    ndim  = len(shape)
    strides = [1] * ndim
    for i in range(1, ndim):
        strides[i] = strides[i - 1] * shape[i - 1]

    if DEBUG:
        print(f"  PARAM @{f.tell():>10d}  shape={shape}  strides={strides}  {label}")

    write_symint(f, -1)
    write_tcapint(f, offset)
    write_tcapint(f, ndim)
    for s, st in zip(shape, strides):
        write_tcapint(f, s)
        write_tcapint(f, st)
    write_storage(f, arr_f)

# ---------------------------------------------------------------------------
# Module writers — one function per ModuleType used in conversion
# ---------------------------------------------------------------------------
def write_linear(f, weight: np.ndarray, bias=None, label=''):
    """weight from HF: (out_features, in_features) — Weed expects (in_features, out_features)."""
    weight = weight.T   # (in_features, out_features)
    in_f, out_f = weight.shape
    if DEBUG:
        print(f" LINEAR @{f.tell():>10d}  in={in_f} out={out_f}  {label}")
    write_module_type(f, ModuleType.LINEAR_T)
    write_tcapint(f, in_f)
    write_tcapint(f, out_f)
    write_parameter(f, weight, label=f'{label}.weight')
    has_bias = bias is not None
    write_bool(f, has_bias)
    if has_bias:
        write_parameter(f, bias, label=f'{label}.bias')

def write_layernorm(f, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5, label=''):
    features = gamma.shape[0]
    if DEBUG:
        print(f" LAYERNORM @{f.tell():>10d}  features={features}  {label}")
    write_module_type(f, ModuleType.LAYERNORM_T)
    write_tcapint(f, features)
    write_real(f, eps)
    # Weed stores gamma/beta as [1, 1, features] (3D) — first two strides are 1
    write_parameter(f, gamma.reshape(1, 1, features), label=f'{label}.gamma')
    write_parameter(f, beta.reshape(1, 1, features),  label=f'{label}.beta')

def write_gelu(f):
    write_module_type(f, ModuleType.GELU_T)

def write_embedding(f, weight: np.ndarray, label=''):
    """weight shape: (num_embeddings, embedding_dim)"""
    num_emb, emb_dim = weight.shape
    if DEBUG:
        print(f" EMBEDDING @{f.tell():>10d}  num_emb={num_emb} emb_dim={emb_dim}  {label}")
    write_module_type(f, ModuleType.EMBEDDING_T)
    write_tcapint(f, num_emb)
    write_tcapint(f, emb_dim)
    write_parameter(f, weight, label=f'{label}.weight')

def write_learned_positional_encoding(f, weight: np.ndarray, label=''):
    """
    weight from HF: (max_len, d_model)
    Weed stores pos_encoding as shape [1, max_len, d_model] (3D) with
    column-major strides [1, 1, max_len] — matching LearnedPositionalEncoding
    constructor: Parameter(init, {1U, max_len, d_model}, dtag)
    """
    max_len, d_model = weight.shape
    if DEBUG:
        print(f" LEARNEDPOS @{f.tell():>10d}  max_len={max_len} d_model={d_model}  {label}")
    write_module_type(f, ModuleType.LEARNED_POSITIONAL_ENCODING_T)
    write_tcapint(f, max_len)
    write_tcapint(f, d_model)
    # Reshape to [1, max_len, d_model] to match Weed's internal representation
    weight_3d = weight.reshape(1, max_len, d_model)
    write_parameter(f, weight_3d, label=f'{label}.weight')

def write_reshape(f, shape, label=''):
    """Write a Reshape module. Use -1 for dynamic/inferred dimensions."""
    if DEBUG:
        print(f" RESHAPE @{f.tell():>10d}  shape={shape}  {label}")
    write_module_type(f, ModuleType.RESHAPE_T)
    write_tcapint(f, len(shape))
    for s in shape:
        write_symint(f, s)  # symint allows -1 for dynamic dims

def write_multihead_attention(f, W_q, b_q, W_k, b_k, W_v, b_v, W_o, b_o,
                               d_model, num_heads, head_dim=None,
                               rope_params=None, mask_val=-1e9, label='attn'):
    if head_dim is None:
        head_dim = d_model // num_heads
    write_module_type(f, ModuleType.MULTIHEAD_ATTENTION_T)
    write_real(f, mask_val)          # mask_val — first, matching deserializer
    write_symint(f, d_model)
    write_symint(f, num_heads)
    write_symint(f, head_dim)
    write_linear(f, W_q, b_q, label=f'{label}.W_q')
    write_linear(f, W_k, b_k, label=f'{label}.W_k')
    write_linear(f, W_v, b_v, label=f'{label}.W_v')
    write_linear(f, W_o, b_o, label=f'{label}.W_o')
    has_rope = rope_params is not None
    write_bool(f, has_rope)
    if has_rope:
        write_rope(f, **rope_params)

def write_transformer_encoder_layer(f, tensors, layer_idx, config, mask_val):
    d_model   = config['d_model']
    d_ff      = config['d_ff']
    num_heads = config['num_heads']
    arch      = config.get('arch', 'gpt2')

    if arch == 'gpt2':
        pfx = f'h.{layer_idx}'
        # GPT-2 fuses Q,K,V into one matrix: (3*d_model, d_model)
        c_attn_w = tensors[f'{pfx}.attn.c_attn.weight']   # (d_model, 3*d_model)
        c_attn_b = tensors.get(f'{pfx}.attn.c_attn.bias')
        c_proj_w = tensors[f'{pfx}.attn.c_proj.weight']
        c_proj_b = tensors.get(f'{pfx}.attn.c_proj.bias')

        # GPT-2 uses Conv1D which stores weights as (in, out) natively.
        # We transpose to (out, in) here to match standard HF Linear convention,
        # then write_linear will transpose again to (in, out) for Weed.
        # Net effect: two transposes = back to original (in, out) GPT-2 layout,
        # which is what Weed expects.
        W_qkv = c_attn_w.T  # (3*d_model, d_model) — now (out, in) HF convention
        W_q = W_qkv[:d_model, :]
        W_k = W_qkv[d_model:2*d_model, :]
        W_v = W_qkv[2*d_model:, :]

        b_q = b_k = b_v = None
        if c_attn_b is not None:
            b_q = c_attn_b[:d_model]
            b_k = c_attn_b[d_model:2*d_model]
            b_v = c_attn_b[2*d_model:]

        W_o = c_proj_w.T   # (d_model, d_model) — now (out, in) HF convention
        b_o = c_proj_b

        ff1_w = tensors[f'{pfx}.mlp.c_fc.weight'].T    # (d_ff, d_model)
        ff1_b = tensors.get(f'{pfx}.mlp.c_fc.bias')
        ff2_w = tensors[f'{pfx}.mlp.c_proj.weight'].T  # (d_model, d_ff)
        ff2_b = tensors.get(f'{pfx}.mlp.c_proj.bias')

        ln1_g = tensors[f'{pfx}.ln_1.weight']
        ln1_b = tensors[f'{pfx}.ln_1.bias']
        ln2_g = tensors[f'{pfx}.ln_2.weight']
        ln2_b = tensors[f'{pfx}.ln_2.bias']
        ln_eps = config.get('layer_norm_epsilon', 1e-5)

    elif arch == 'bert':
        pfx = f'encoder.layer.{layer_idx}'
        W_q = tensors[f'{pfx}.attention.self.query.weight']
        b_q = tensors.get(f'{pfx}.attention.self.query.bias')
        W_k = tensors[f'{pfx}.attention.self.key.weight']
        b_k = tensors.get(f'{pfx}.attention.self.key.bias')
        W_v = tensors[f'{pfx}.attention.self.value.weight']
        b_v = tensors.get(f'{pfx}.attention.self.value.bias')
        W_o = tensors[f'{pfx}.attention.output.dense.weight']
        b_o = tensors.get(f'{pfx}.attention.output.dense.bias')

        ff1_w = tensors[f'{pfx}.intermediate.dense.weight']
        ff1_b = tensors.get(f'{pfx}.intermediate.dense.bias')
        ff2_w = tensors[f'{pfx}.output.dense.weight']
        ff2_b = tensors.get(f'{pfx}.output.dense.bias')

        ln1_g = tensors[f'{pfx}.attention.output.LayerNorm.weight']
        ln1_b = tensors[f'{pfx}.attention.output.LayerNorm.bias']
        ln2_g = tensors[f'{pfx}.output.LayerNorm.weight']
        ln2_b = tensors[f'{pfx}.output.LayerNorm.bias']
        ln_eps = config.get('layer_norm_eps', 1e-12)
    else:
        raise ValueError(f"Unknown arch: {arch}")

    write_module_type(f, ModuleType.TRANSFORMER_ENCODER_LAYER_T)
    write_tcapint(f, d_model)
    write_tcapint(f, d_ff)
    write_tcapint(f, num_heads)
    write_multihead_attention(f, W_q, b_q, W_k, b_k, W_v, b_v, W_o, b_o,
                               d_model, num_heads, None, None, mask_val)
    write_linear(f, ff1_w, ff1_b)
    write_linear(f, ff2_w, ff2_b)
    write_layernorm(f, ln1_g, ln1_b, ln_eps)
    write_layernorm(f, ln2_g, ln2_b, ln_eps)
    write_gelu(f)  # activation — adjust if arch uses ReLU etc.

def write_qwen_transformer_layer(f, tensors, layer_idx, config, mask_val):
    """
    Qwen2/Qwen3 decoder layer. Key differences from GPT-2/BERT:
    - Separate q_proj, k_proj, v_proj (GQA: k/v heads may be fewer)
    - SwiGLU FFN: gate_proj + up_proj → silu(gate) * up → down_proj
    - RMSNorm instead of LayerNorm (no beta/bias — we write zeros)
    - Rotary embeddings are baked into weights; no explicit pos encoding module
    """
    d_model      = config['d_model']
    d_ff         = config['d_ff']
    num_heads    = config['num_heads']
    num_kv_heads = config['num_kv_heads']
    eps          = config.get('layer_norm_eps', 1e-6)
    rope_base    = config.get('rope_theta', 10000.0)
    max_seq_len  = config.get('max_position_embeddings', 2048)
    mask_val     = config.get('mask_val', -1e9)
    head_dim     = d_model // num_heads
    pfx          = f'model.layers.{layer_idx}'

    W_q = tensors[f'{pfx}.self_attn.q_proj.weight']
    b_q = tensors.get(f'{pfx}.self_attn.q_proj.bias')
    W_k = tensors[f'{pfx}.self_attn.k_proj.weight']
    b_k = tensors.get(f'{pfx}.self_attn.k_proj.bias')
    W_v = tensors[f'{pfx}.self_attn.v_proj.weight']
    b_v = tensors.get(f'{pfx}.self_attn.v_proj.bias')
    W_o = tensors[f'{pfx}.self_attn.o_proj.weight']
    b_o = tensors.get(f'{pfx}.self_attn.o_proj.bias')

    gate_w = tensors[f'{pfx}.mlp.gate_proj.weight']
    up_w   = tensors[f'{pfx}.mlp.up_proj.weight']
    down_w = tensors[f'{pfx}.mlp.down_proj.weight']

    ln1_g = tensors[f'{pfx}.input_layernorm.weight']
    ln2_g = tensors[f'{pfx}.post_attention_layernorm.weight']

    write_module_type(f, ModuleType.QWEN_DECODER_LAYER_T)
    write_tcapint(f, d_model)
    write_tcapint(f, num_heads)
    write_tcapint(f, num_kv_heads)
    write_multihead_attention(f, W_q, b_q, W_k, b_k, W_v, b_v, W_o, b_o,
                               d_model, num_heads, None, None, mask_val)
    write_rope(f, head_dim, max_seq_len, rope_base)
    write_swiglu(f, gate_w, up_w, down_w, label=f'{pfx}.mlp')
    write_rmsnorm(f, ln1_g, eps, label=f'{pfx}.input_layernorm')
    write_rmsnorm(f, ln2_g, eps, label=f'{pfx}.post_attention_layernorm')

def write_rmsnorm(f, gamma: np.ndarray, eps: float = 1e-6, label=''):
    hidden_size = gamma.shape[0]
    if DEBUG:
        print(f" RMSNORM @{f.tell():>10d}  hidden_size={hidden_size}  {label}")
    write_module_type(f, ModuleType.RMS_NORM_T)
    write_symint(f, -1)          # axis = -1
    write_tcapint(f, hidden_size)
    # weight: shape [hidden_size], stride [1], initialised to gamma
    write_parameter(f, gamma.reshape(hidden_size), label=f'{label}.weight')

def write_rope(f, head_dim: int, max_seq_len: int = 2048,
               base: float = 10000.0, label=''):
    if DEBUG:
        print(f" ROPE @{f.tell():>10d}  head_dim={head_dim}  "
              f"max_seq_len={max_seq_len}  base={base}  {label}")
    write_module_type(f, ModuleType.ROPE_T)
    write_tcapint(f, head_dim)
    write_tcapint(f, max_seq_len)
    write_real(f, base)
    # cos/sin tables are recomputed on load — nothing else to write

def write_swiglu(f, gate_w: np.ndarray, up_w: np.ndarray,
                 down_w: np.ndarray, label=''):
    hidden_size        = gate_w.shape[1]   # (d_ff, d_model) → in = d_model
    intermediate_size  = gate_w.shape[0]   # out = d_ff
    if DEBUG:
        print(f" SWIGLU @{f.tell():>10d}  "
              f"hidden={hidden_size} intermediate={intermediate_size}  {label}")
    write_module_type(f, ModuleType.SWIGLU_T)
    write_tcapint(f, hidden_size)
    write_tcapint(f, intermediate_size)
    write_linear(f, gate_w, None, label=f'{label}.gate_proj')
    write_linear(f, up_w,   None, label=f'{label}.up_proj')
    write_linear(f, down_w, None, label=f'{label}.down_proj')

def write_qwen_model(f, tensors, config):
    n_layer     = config['n_layer']
    d_model     = config['d_model']
    eps         = config.get('layer_norm_eps', 1e-6)

    # embed_tokens + N decoder layers + final RMSNorm + lm_head
    n_modules = 1 + n_layer + 1 + 1
    write_module_type(f, ModuleType.SEQUENTIAL_T)
    write_tcapint(f, n_modules)

    write_embedding(f, tensors['model.embed_tokens.weight'],
                    label='embed_tokens')

    for i in range(n_layer):
        write_qwen_transformer_layer(f, tensors, i, config)

    write_rmsnorm(f, tensors['model.norm.weight'], eps, label='model.norm')

    lm_w = tensors.get('lm_head.weight',
                        tensors['model.embed_tokens.weight'])
    write_linear(f, lm_w, None, label='lm_head')


# ---------------------------------------------------------------------------
# Top-level sequential model writers per architecture
# ---------------------------------------------------------------------------
def write_gpt2_model(f, tensors, config):
    n_layer   = config['n_layer']
    d_model   = config['d_model']
    mask_val   = config['mask_val']

    # token_emb + reshape + pos_emb + N layers + final_ln + lm_head
    n_modules = 3 + n_layer + 2
    write_module_type(f, ModuleType.SEQUENTIAL_T)
    write_tcapint(f, n_modules)

    # Token embedding: [seq_len] → [seq_len, emb_dim]
    write_embedding(f, tensors['wte.weight'], label='wte')

    # Reshape to 3D [1, seq_len, emb_dim] so LearnedPositionalEncoding
    # receives (B, T, d_model) as it expects via sh[1] = T
    write_reshape(f, [1, -1, d_model], label='unsqueeze_batch')

    # Learned positional encoding
    write_learned_positional_encoding(f, tensors['wpe.weight'], label='wpe')

    # Transformer layers
    cfg = dict(config, arch='gpt2')
    for i in range(n_layer):
        write_transformer_encoder_layer(f, tensors, i, cfg, mask_val)

    # Final layer norm
    write_layernorm(f, tensors['ln_f.weight'], tensors['ln_f.bias'],
                    config.get('layer_norm_epsilon', 1e-5), label='ln_f')

    # LM head (weight-tied to token embedding in standard GPT-2)
    lm_w = tensors.get('lm_head.weight', tensors['wte.weight'])
    write_linear(f, lm_w, None, label='lm_head')

def write_bert_model(f, tensors, config):
    n_layer = config['num_hidden_layers']
    d_model = config['hidden_size']
    mask_val   = config['mask_val']

    n_modules = 3 + n_layer + 1  # word_emb + pos_emb + tok_type_emb + layers + pooler
    write_module_type(f, ModuleType.SEQUENTIAL_T)
    write_tcapint(f, n_modules)

    write_embedding(f, tensors['embeddings.word_embeddings.weight'])
    write_learned_positional_encoding(
        f, tensors['embeddings.position_embeddings.weight'])
    write_embedding(f, tensors['embeddings.token_type_embeddings.weight'])

    cfg = dict(config, arch='bert')
    for i in range(n_layer):
        write_transformer_encoder_layer(f, tensors, i, cfg)

    # Pooler
    write_linear(f, tensors['pooler.dense.weight'],
                 tensors.get('pooler.dense.bias'))

# ---------------------------------------------------------------------------
# Config normalisation — map HF config.json fields to unified keys
# ---------------------------------------------------------------------------
def normalise_config(raw, arch, mask_val):
    if arch == 'gpt2':
        return {
            'arch':                 'gpt2',
            'mask_val':             mask_val,
            'n_layer':              raw['n_layer'],
            'd_model':              raw['n_embd'],
            'd_ff':                 raw.get('n_inner') or 4 * raw['n_embd'],
            'num_heads':            raw['n_head'],
            'layer_norm_epsilon':   raw.get('layer_norm_epsilon', 1e-5),
        }
    elif arch == 'bert':
        return {
            'arch':                 'bert',
            'mask_val':             mask_val,
            'n_layer':              raw['num_hidden_layers'],
            'd_model':              raw['hidden_size'],
            'd_ff':                 raw['intermediate_size'],
            'num_heads':            raw['num_attention_heads'],
            'layer_norm_eps':       raw.get('layer_norm_eps', 1e-12),
        }
    elif arch == 'qwen':
        return {
            'arch':                    'qwen',
            'mask_val':                mask_val,
            'n_layer':                 raw['num_hidden_layers'],
            'd_model':                 raw['hidden_size'],
            'd_ff':                    raw['intermediate_size'],
            'num_heads':               raw['num_attention_heads'],
            'num_kv_heads':            raw.get('num_key_value_heads',
                                               raw['num_attention_heads']),
            'layer_norm_eps':          raw.get('rms_norm_eps', 1e-6),
            'rope_theta':              raw.get('rope_theta', 10000.0),
            'max_position_embeddings': raw.get('max_position_embeddings', 2048),
        }
    else:
        raise ValueError(f"Unsupported arch: {arch}")

# ---------------------------------------------------------------------------
# Safetensors loader — returns dict of key → np.ndarray (float32)
# ---------------------------------------------------------------------------
def load_safetensors(model_dir: Path):
    # If a sharding index exists, use it to find the actual shard files.
    index_path = model_dir / 'model.safetensors.index.json'
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        # weight_map: { tensor_name: shard_filename }
        shard_files = sorted(set(index['weight_map'].values()))
        st_files = [model_dir / s for s in shard_files]
        print(f"Sharded model: {len(st_files)} shard(s) via index.")
    else:
        # Single-file or unindexed multi-file: load every .safetensors file
        # that is NOT an index file (some tools emit foo.safetensors.index.json
        # with a .safetensors extension by mistake).
        st_files = sorted(
            p for p in model_dir.glob('*.safetensors')
            if 'index' not in p.name.lower()
        )
        if not st_files:
            raise FileNotFoundError(f"No .safetensors files found in {model_dir}")

    tensors = {}
    for st_path in st_files:
        print(f"  Loading {st_path.name} ...")
        try:
            with safe_open(str(st_path), framework='numpy') as f:
                for key in f.keys():
                    tensors[key] = f.get_tensor(key).astype(np.float32)
        except Exception as e:
            print(f"  WARNING: skipping {st_path.name}: {e}")
            continue

    if not tensors:
        raise RuntimeError("No tensors loaded — check model directory contents.")

    print(f"Loaded {len(tensors)} tensors from {len(st_files)} safetensors file(s).")
    return tensors

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Convert HuggingFace safetensors model to Weed binary format.')
    parser.add_argument('--list_keys', required=False, action='store_true',
                        help='List tensor shapes and types')
    parser.add_argument('--model_dir', required=True,
                        help='Directory containing model.safetensors and config.json')
    parser.add_argument('--output', default='model.weed',
                        help='Output filename (default: model.weed)')
    parser.add_argument('--arch', default='auto',
                        choices=['auto', 'gpt2', 'bert', 'qwen'],
                        help='Model architecture (default: auto-detect from config.json)')
    parser.add_argument('--mask_val', type=float, default=-1e9,
                    help='Causal attention mask value (default: -1e9 for HF models)')
    parser.add_argument('--debug', action='store_true',
                        help='Print file offsets and shapes for every parameter written')
    args = parser.parse_args()

    global DEBUG
    if args.debug:
        DEBUG = True

    model_dir = Path(args.model_dir)

    tensors = load_safetensors(model_dir)

    if args.list_keys:
        for k, v in sorted(tensors.items()):
            print(f"  {k:60s} {str(v.shape):30s} {v.dtype}")
        return

    # Load config.json
    config_path = model_dir / 'config.json'
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {model_dir}")
    with open(config_path) as cf:
        raw_config = json.load(cf)

    # Auto-detect architecture
    arch = args.arch
    if arch == 'auto':
        model_type = raw_config.get('model_type', '').lower()
        if 'gpt2' in model_type:
            arch = 'gpt2'
        elif 'bert' in model_type:
            arch = 'bert'
        elif 'qwen' in model_type:
            arch = 'qwen'
        else:
            print(f"Could not auto-detect architecture from model_type='{model_type}'.")
            print("Known types: gpt2, bert, qwen")
            print("Use --list_keys to inspect tensor names and --arch to specify manually.")
            sys.exit(1)

    print(f"Architecture: {arch}")
    config = normalise_config(raw_config, arch, args.mask_val)
    print(f"Config: {json.dumps(config, indent=2)}")

    output_path = Path(args.output)
    with open(output_path, 'wb') as f:
        if arch == 'gpt2':
            write_gpt2_model(f, tensors, config)
        elif arch == 'bert':
            write_bert_model(f, tensors, config)
        elif arch == 'qwen':
            write_qwen_model(f, tensors, config)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Written: {output_path}  ({size_mb:.1f} MB)")

if __name__ == '__main__':
    main()
