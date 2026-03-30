#!/usr/bin/env python3
"""
weed_gpt2_chat.py — Minimal GPT-2 text generation using Weed inference.

(C) Daniel Strano and the Qrack contributors 2026.

This file was produced almost in its entirety, verbatim, by (Anthropic) Claude.

Use of this source code is governed by an MIT-style license that can be
found in the LICENSE file or at https://opensource.org/licenses/MIT.

Usage:
    python3 weed_gpt2_chat.py --model <path_to.weed> --tokenizer <hf_model_dir>
    python3 weed_gpt2_chat.py --model gpt2.weed --tokenizer ./gpt2 --prompt "Hello"
"""

import argparse
import math
import random
import sys

# ---------------------------------------------------------------------------
# Tokenizer — uses HuggingFace tokenizers (no PyTorch required)
# ---------------------------------------------------------------------------
try:
    from tokenizers import Tokenizer
except ImportError:
    print("Missing dependency: pip install tokenizers")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Weed imports — adjust path if weed_loader is not installed as a package
# ---------------------------------------------------------------------------
try:
    from weed_loader.weed_module import WeedModule
    from weed_loader.weed_tensor import WeedTensor
    from weed_loader.dtype import DType
except ImportError:
    print("weed_loader not found. Ensure it is installed or on PYTHONPATH.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------
def softmax(logits: list[float]) -> list[float]:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]


def top_p_sample(logits: list[float], top_p: float = 0.9,
                 temperature: float = 1.0) -> int:
    """Nucleus (top-p) sampling."""
    if temperature != 1.0:
        logits = [l / temperature for l in logits]
    probs = softmax(logits)
    # Sort descending by probability
    indexed = sorted(enumerate(probs), key=lambda x: -x[1])
    cumulative = 0.0
    nucleus = []
    for idx, p in indexed:
        nucleus.append((idx, p))
        cumulative += p
        if cumulative >= top_p:
            break
    # Renormalise nucleus
    total = sum(p for _, p in nucleus)
    r = random.random() * total
    cumulative = 0.0
    for idx, p in nucleus:
        cumulative += p
        if r <= cumulative:
            return idx
    return nucleus[-1][0]


def greedy(logits: list[float]) -> int:
    return logits.index(max(logits))


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def generate(model: WeedModule,
             input_ids: list[int],
             max_new_tokens: int = 100,
             temperature: float = 1.0,
             top_p: float = 0.9,
             greedy_decode: bool = False,
             eos_token_id: int = 50256) -> list[int]:
    """
    Autoregressive generation loop.

    Weed's Sequential model processes the full token sequence each forward
    pass and returns logits of shape (seq_len, vocab_size) or (vocab_size,).
    We take logits at the last position for next-token prediction.
    """
    tokens = list(input_ids)

    model.set_max_kv_seq_len(max_new_tokens)

    for _ in range(max_new_tokens):
        # Build INT tensor: shape [seq_len]
        t = WeedTensor(
            data=tokens,
            shape=[len(tokens)],
            stride=[1],
            dtype=DType.INT,
            offset=0
        )

        result = model.forward(t)

        # result.data is flat; result.shape tells us the layout
        # Expected: (seq_len, vocab_size) flattened, or just (vocab_size,)
        shape = list(result.shape)
        data  = list(result.data)

        if len(shape) == 3:
            # [1, seq_len, vocab_size] column-major
            _, seq_len, vocab_size = shape
            # In column-major the last seq position is at index (seq_len-1) across
            # the second dimension — stride[1] = shape[0] = 1, stride[2] = seq_len
            # So logit[0, t, v] = data[0 + t*1 + v*seq_len]
            last_logits = [data[(seq_len - 1) * 1 + v * seq_len] 
                           for v in range(vocab_size)]
        elif len(shape) == 2:
            # [seq_len, vocab_size] — take last position col-major
            seq_len, vocab_size = shape
            last_logits = [data[(seq_len - 1) + v * seq_len]
                           for v in range(vocab_size)]
        elif len(shape) == 1:
            last_logits = data
        else:
            raise RuntimeError(f"Unexpected output shape: {shape}")

        if greedy_decode or temperature == 0.0:
            next_token = greedy(last_logits)
        else:
            next_token = top_p_sample(last_logits, top_p=top_p,
                                      temperature=temperature)

        tokens.append(next_token)
            
        if next_token == eos_token_id:
            break

    model.reset_kv_cache()  # always reset, even on exception

    return tokens[len(input_ids):]  # return only newly generated tokens


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def load_tokenizer(tokenizer_dir: str) -> Tokenizer:
    """Load GPT-2 BPE tokenizer from a HuggingFace model directory."""
    import os
    # Try tokenizer.json first (fast tokenizer)
    tok_json = os.path.join(tokenizer_dir, 'tokenizer.json')
    if os.path.exists(tok_json):
        return Tokenizer.from_file(tok_json)
    # Fall back to pretrained name if directory doesn't have tokenizer.json
    return Tokenizer.from_pretrained(tokenizer_dir)


def main():
    parser = argparse.ArgumentParser(
        description='GPT-2 text generation with Weed inference engine.')
    parser.add_argument('--model', required=True,
                        help='Path to .weed model file')
    parser.add_argument('--tokenizer', required=True,
                        help='HuggingFace model dir containing tokenizer.json, '
                             'or HF repo id (e.g. gpt2)')
    parser.add_argument('--prompt', default='Hello',
                        help='Input prompt (default: "Hello")')
    parser.add_argument('--max_new_tokens', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--greedy', action='store_true',
                        help='Use greedy decoding instead of sampling')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    print(f"Loading tokenizer from: {args.tokenizer}")
    tokenizer = load_tokenizer(args.tokenizer)

    print(f"Loading model from: {args.model}")
    model = WeedModule(args.model)
    print("Model loaded.")

    # Encode prompt
    encoding = tokenizer.encode(args.prompt)
    input_ids = encoding.ids
    print(f"Prompt ({len(input_ids)} tokens): {args.prompt!r}")
    print()

    # Generate
    new_tokens = generate(
        model=model,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        greedy_decode=args.greedy,
    )

    # Decode
    full_ids = input_ids + new_tokens
    output_text = tokenizer.decode(full_ids)

    print("=" * 60)
    print(output_text)
    print("=" * 60)
    print(f"\n({len(new_tokens)} new tokens generated)")

    # Interactive mode if no prompt was specified on CLI
    if len(sys.argv) == 3:  # only --model and --tokenizer given
        print("\nEntering interactive mode. Ctrl+C to exit.\n")
        while True:
            try:
                prompt = input("You: ")
            except (KeyboardInterrupt, EOFError):
                break
            if not prompt.strip():
                continue
            enc = tokenizer.encode(prompt)
            new_tok = generate(
                model=model,
                input_ids=enc.ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                greedy_decode=args.greedy,
            )
            print("GPT-2:", tokenizer.decode(enc.ids + new_tok))
            print()


if __name__ == '__main__':
    main()
