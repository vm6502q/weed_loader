#!/usr/bin/env python3
"""
check_hf_parity.py — Check Weed inference against the HuggingFace reference.

(C) Daniel Strano and the Qrack contributors 2026.

This file was produced almost in its entirety, verbatim, by (Anthropic) Claude.

Use of this source code is governed by an MIT-style license that can be
found in the LICENSE file or at https://opensource.org/licenses/MIT.

For each model, this converts the checkpoint with hf_to_weed.py, runs the same
token IDs through Weed and through the HuggingFace (PyTorch) implementation,
and compares the logits at every position, three ways:
  1. the whole sequence in one pass,
  2. a prompt, then one token at a time through the KV cache (as in text
     generation), and
  3. several tokens at a time, continuing from a non-empty KV cache.

Usage:
    # Small randomly-initialized GPT-2 and Qwen2 models (nothing to download)
    python3 check_hf_parity.py

    # A real checkpoint directory (config.json + model.safetensors)
    python3 check_hf_parity.py --model_dir ./gpt2

Needs: pip install torch transformers safetensors numpy
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

try:
    import numpy as np
    import torch
    import transformers
    from safetensors.numpy import save_file
except ImportError:
    print("Missing dependencies. Install with:")
    print("    pip install torch transformers safetensors numpy")
    sys.exit(1)

try:
    from weed_loader.weed_module import WeedModule
    from weed_loader.weed_tensor import WeedTensor
    from weed_loader.dtype import DType
except ImportError:
    print("weed_loader not found. Ensure it is installed or on PYTHONPATH.")
    sys.exit(1)

HF_TO_WEED = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'hf_to_weed.py')


# ---------------------------------------------------------------------------
# Reference models
# ---------------------------------------------------------------------------
def randomize(model, scale):
    """
    The default initialization (std 0.02, unit norms, zero biases) is too tame
    to expose a wrong attention layout, so spread the weights out.
    """
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() == 1 and ('ln_' in name or 'norm' in name) \
                    and name.endswith('weight'):
                p.copy_(1.0 + 0.3 * torch.randn_like(p))
            elif 'bias' in name:
                p.copy_(0.2 * torch.randn_like(p))
            else:
                p.mul_(scale)


def save_checkpoint(model, config, out_dir, strip_prefix=None, skip=()):
    sd = {}
    for k, v in model.state_dict().items():
        if any(k.endswith(s) for s in skip):
            continue
        if strip_prefix and k.startswith(strip_prefix):
            k = k[len(strip_prefix):]
        sd[k] = v.detach().numpy().astype(np.float32)
    save_file(sd, os.path.join(out_dir, 'model.safetensors'))
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump(config.to_dict(), f)


def make_random_gpt2(out_dir, seed):
    torch.manual_seed(seed)
    config = transformers.GPT2Config(
        vocab_size=97, n_positions=40, n_embd=48, n_layer=3, n_head=4,
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
    model = transformers.GPT2LMHeadModel(config).eval()
    randomize(model, 6.0)
    # Same key layout as the "gpt2" repository: no "transformer." prefix, and
    # no separate (tied) lm_head or attention mask buffers.
    save_checkpoint(model, config, out_dir, strip_prefix='transformer.',
                    skip=('.attn.bias', '.masked_bias', 'lm_head.weight'))
    return model


def make_random_qwen(out_dir, seed):
    torch.manual_seed(seed)
    config = transformers.Qwen2Config(
        vocab_size=97, hidden_size=48, intermediate_size=80,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, rms_norm_eps=1e-6, attention_dropout=0.0,
        tie_word_embeddings=False)
    model = transformers.Qwen2ForCausalLM(config).eval()
    randomize(model, 6.0)
    save_checkpoint(model, config, out_dir)
    return model


# ---------------------------------------------------------------------------
# Weed
# ---------------------------------------------------------------------------
def weed_forward(model, ids):
    """Returns logits as a (len(ids), vocab_size) array."""
    t = WeedTensor(data=list(ids), shape=[len(ids)], stride=[1],
                   dtype=DType.INT, offset=0)
    r = model.forward(t)
    shape = list(r.shape)
    size = int(np.prod(shape))
    # Weed tensors are column-major
    out = np.array(r.data[:size]).reshape(shape, order='F')
    return out.reshape(-1, shape[-1])


def weed_chunked(model, ids, chunks):
    model.reset_kv_cache()
    rows = []
    start = 0
    for n in chunks:
        out = weed_forward(model, ids[start:start + n])
        if out.shape[0] != n:
            raise RuntimeError(f"Weed returned {out.shape[0]} positions for "
                               f"{n} tokens")
        rows.append(out)
        start += n
    model.reset_kv_cache()
    return np.concatenate(rows, 0)


def check(name, hf_model, model_dir, ids, tol):
    with torch.no_grad():
        ref = hf_model(torch.tensor([ids])).logits[0].float().numpy()
    scale = float(np.abs(ref).mean())

    weed_file = os.path.join(model_dir, 'parity_check.weed')
    subprocess.run([sys.executable, HF_TO_WEED, '--model_dir', model_dir,
                    '--output', weed_file],
                   check=True, stdout=subprocess.DEVNULL)
    model = WeedModule(weed_file)

    T = len(ids)
    k = max(1, T // 3)
    runs = [
        ('whole sequence', [T]),
        ('prompt, then token by token', [k] + [1] * (T - k)),
        ('several tokens at a time', [k, T - 2 * k, k]),
    ]

    ok = True
    print(f"{name}: {T} tokens, mean |logit| = {scale:.3g}")
    for label, chunks in runs:
        out = weed_chunked(model, ids, chunks)
        err = float(np.abs(out - ref).max())
        agree = float(np.mean(out.argmax(-1) == ref.argmax(-1)))
        passed = (err <= tol * max(scale, 1.0)) and (agree == 1.0)
        ok = ok and passed
        print(f"  {'PASS' if passed else 'FAIL'}  {label:30s} "
              f"max |Weed - HF| = {err:.3g}, "
              f"next-token agreement = {100 * agree:.0f}%")

    del model
    os.remove(weed_file)
    return ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Check Weed inference against the HuggingFace reference.')
    parser.add_argument('--model_dir', default=None,
                        help='HuggingFace checkpoint directory (default: use '
                             'small randomly-initialized GPT-2 and Qwen2)')
    parser.add_argument('--num_tokens', type=int, default=12,
                        help='Length of the random token sequence')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--tol', type=float, default=2e-3,
                        help='Allowed max logit error, relative to the mean '
                             'logit magnitude')
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    ok = True

    if args.model_dir is not None:
        hf_model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model_dir).eval()
        vocab = hf_model.config.vocab_size
        ids = rng.integers(0, vocab, size=args.num_tokens).tolist()
        ok = check(args.model_dir, hf_model, args.model_dir, ids, args.tol)
    else:
        for name, make in (('GPT-2 (random weights)', make_random_gpt2),
                           ('Qwen2 (random weights)', make_random_qwen)):
            with tempfile.TemporaryDirectory() as d:
                hf_model = make(d, args.seed)
                ids = rng.integers(0, hf_model.config.vocab_size,
                                   size=args.num_tokens).tolist()
                ok = check(name, hf_model, d, ids, args.tol) and ok

    print("All checks passed." if ok else "Some checks FAILED.")
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
