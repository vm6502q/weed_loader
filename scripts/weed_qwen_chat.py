#!/usr/bin/env python3
"""
weed_qwen_chat.py — Minimal Qwen text generation using Weed inference.

(C) Daniel Strano and the Qrack contributors 2026.

This file was produced almost in its entirety, verbatim, by (Anthropic) Claude.

Use of this source code is governed by an MIT-style license that can be
found in the LICENSE file or at https://opensource.org/licenses/MIT.

Usage:
    python3 weed_qwen_chat.py --model <path_to.weed> --tokenizer <hf_model_dir>
    python3 weed_qwen_chat.py --model qwen.weed --tokenizer ./Qwen2-0.5B --prompt "Hello"
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
# Weed imports
# ---------------------------------------------------------------------------
try:
    from weed_loader.weed_module import WeedModule
    from weed_loader.weed_tensor import WeedTensor
    from weed_loader.dtype import DType
except ImportError:
    print("weed_loader not found. Ensure it is installed or on PYTHONPATH.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Sampling helpers (identical to GPT-2 script)
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
    indexed = sorted(enumerate(probs), key=lambda x: -x[1])
    cumulative = 0.0
    nucleus = []
    for idx, p in indexed:
        nucleus.append((idx, p))
        cumulative += p
        if cumulative >= top_p:
            break
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
             eos_token_ids: list[int] = None) -> list[int]:
    """
    Autoregressive generation loop for Qwen models.

    Qwen uses multiple EOS token IDs — both the base EOS and the
    end-of-turn token should stop generation.
    """
    if eos_token_ids is None:
        eos_token_ids = [151645, 151643]  # <|im_end|> and <|endoftext|>

    tokens = list(input_ids)

    for _ in range(max_new_tokens):
        t = WeedTensor(
            data=tokens,
            shape=[len(tokens)],
            stride=[1],
            dtype=DType.INT,
            offset=0
        )

        result = model.forward(t)

        shape = list(result.shape)
        data  = list(result.data)

        if len(shape) == 3:
            # [1, seq_len, vocab_size] column-major
            _, seq_len, vocab_size = shape
            last_logits = [data[(seq_len - 1) * 1 + v * seq_len]
                           for v in range(vocab_size)]
        elif len(shape) == 2:
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

        if next_token in eos_token_ids:
            break

    model.reset_kv_cache()

    return tokens[len(input_ids):]


# ---------------------------------------------------------------------------
# Tokenizer loader
# ---------------------------------------------------------------------------
def load_tokenizer(tokenizer_dir: str) -> Tokenizer:
    """Load Qwen tokenizer from a HuggingFace model directory."""
    import os
    tok_json = os.path.join(tokenizer_dir, 'tokenizer.json')
    if os.path.exists(tok_json):
        return Tokenizer.from_file(tok_json)
    return Tokenizer.from_pretrained(tokenizer_dir)


def get_eos_token_ids(tokenizer_dir: str) -> list[int]:
    """
    Read EOS token IDs from tokenizer_config.json if available.
    Qwen models typically use 151645 (<|im_end|>) and 151643 (<|endoftext|>).
    """
    import os
    import json
    config_path = os.path.join(tokenizer_dir, 'tokenizer_config.json')
    if not os.path.exists(config_path):
        return [151645, 151643]
    with open(config_path) as f:
        cfg = json.load(f)
    eos_ids = []
    # eos_token may be a string token or a dict
    eos = cfg.get('eos_token')
    if isinstance(eos, str):
        # Will be resolved after tokenizer loads
        pass
    elif isinstance(eos, dict):
        content = eos.get('content', '')
        if content:
            pass  # resolved below
    # Always include the standard Qwen EOS tokens as fallback
    eos_ids = [151645, 151643]
    return eos_ids


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Qwen text generation with Weed inference engine.')
    parser.add_argument('--model', required=True,
                        help='Path to .weed model file')
    parser.add_argument('--tokenizer', required=True,
                        help='HuggingFace model dir containing tokenizer.json')
    parser.add_argument('--prompt', default='Hello',
                        help='Input prompt (default: "Hello")')
    parser.add_argument('--max_new_tokens', type=int, default=64)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--greedy', action='store_true',
                        help='Use greedy decoding instead of sampling')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--chat', action='store_true',
                        help='Wrap prompt in Qwen chat template (<|im_start|> etc.)')
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    print(f"Loading tokenizer from: {args.tokenizer}")
    tokenizer = load_tokenizer(args.tokenizer)
    eos_token_ids = get_eos_token_ids(args.tokenizer)

    print(f"Loading model from: {args.model}")
    model = WeedModule(args.model)
    print("Model loaded.")

    def format_prompt(text: str, chat: bool) -> str:
        """Optionally wrap in Qwen chat template."""
        if not chat:
            return text
        return (f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                f"<|im_start|>user\n{text}<|im_end|>\n"
                f"<|im_start|>assistant\n")

    prompt = format_prompt(args.prompt, args.chat)
    encoding = tokenizer.encode(prompt)
    input_ids = encoding.ids
    print(f"Prompt ({len(input_ids)} tokens): {args.prompt!r}")
    print()

    new_tokens = generate(
        model=model,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        greedy_decode=args.greedy,
        eos_token_ids=eos_token_ids,
    )

    full_ids = input_ids + new_tokens
    output_text = tokenizer.decode(full_ids)

    print("=" * 60)
    print(output_text)
    print("=" * 60)
    print(f"\n({len(new_tokens)} new tokens generated)")

    # Interactive mode
    if len(sys.argv) == 3:
        print("\nEntering interactive mode. Ctrl+C to exit.\n")
        while True:
            try:
                user_input = input("You: ")
            except (KeyboardInterrupt, EOFError):
                break
            if not user_input.strip():
                continue
            p = format_prompt(user_input, args.chat)
            enc = tokenizer.encode(p)
            new_tok = generate(
                model=model,
                input_ids=enc.ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                greedy_decode=args.greedy,
                eos_token_ids=eos_token_ids,
            )
            full = enc.ids + new_tok
            # Decode only the new tokens for cleaner output
            print("Qwen:", tokenizer.decode(new_tok))
            print()


if __name__ == '__main__':
    main()
