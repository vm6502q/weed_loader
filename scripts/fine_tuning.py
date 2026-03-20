import argparse
import shutil
import sys

try:
    from tokenizers import Tokenizer
except ImportError:
    print("Missing dependency: pip install tokenizers")
    sys.exit(1)

try:
    from weed_loader.weed_module import WeedModule
    from weed_loader.weed_tensor import WeedTensor
    from weed_loader.dtype import DType
except ImportError:
    print("weed_loader not found. Ensure it is installed or on PYTHONPATH.")
    sys.exit(1)

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
        description='GPT-2 fine tuning with Weed inference engine.')
    parser.add_argument('--model', required=True,
                        help='Path to .weed model file')
    parser.add_argument('--tokenizer', required=True,
                        help='HuggingFace model dir containing tokenizer.json, '
                             'or HF repo id (e.g. gpt2)')
    parser.add_argument('--prompt', required=True,
                        help='Input prompt')
    parser.add_argument('--completion', required=True,
                        help='Target for next tokens after input prompt')
    parser.add_argument('--rate', type=float, default=1e-5,
                        help='Learning rate')
    args = parser.parse_args()

    print(f"Backing up model to: {args.model}.bak")
    shutil.copy2(args.model, args.model + ".bak")
    print("Model backed up.")

    print(f"Loading tokenizer from: {args.tokenizer}")
    tokenizer = load_tokenizer(args.tokenizer)

    print(f"Loading model from: {args.model}")
    model = WeedModule(args.model)
    print("Model loaded.")

    # prompt = "Barb (Elara), we're fixing it. (You're on 'Weed.')"
    # completion = " said the navigator (on a ship of fools), the dignitary from set theory. (We were family, eventually.)"

    full_text = args.prompt + args.completion
    ids = tokenizer.encode(full_text).ids

    print(f"Full sequence: {len(ids)} tokens ({len(ids)-1} training pairs)")

    # input is ids[:-1], targets are ids[1:]
    print("Training...")
    model.train_step(ids[:-1], ids[1:], args.rate)
    print("Training complete.")

    print("Saving model...")
    model.save(args.model)
    print("Saved.")

if __name__ == '__main__':
    main()
