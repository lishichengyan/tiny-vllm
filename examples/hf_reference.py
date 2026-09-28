"""Run the HuggingFace reference implementation of the model.

Works before any milestone is implemented -- it never touches tiny_vllm's model code.
Use it to see what the target output looks like and to sanity check the environment.

    python examples/hf_reference.py --prompt "The capital of France is" --max-new-tokens 20
"""

import argparse
import time

import torch

from tiny_vllm.loader import DEFAULT_MODEL, load_hf_model, load_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=20)
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.model)
    model = load_hf_model(args.model)
    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids
    print(f"model: {args.model}")
    print(f"prompt tokens: {input_ids[0].tolist()}")

    with torch.no_grad():
        t0 = time.perf_counter()
        logits = model(input_ids).logits
        t1 = time.perf_counter()
    print(f"logits shape: {tuple(logits.shape)}   forward: {(t1 - t0) * 1000:.1f} ms")
    print(f"next token (greedy): {tokenizer.decode(logits[0, -1].argmax())!r}")

    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    print("generated:", repr(tokenizer.decode(out[0, input_ids.shape[1] :])))


if __name__ == "__main__":
    main()
