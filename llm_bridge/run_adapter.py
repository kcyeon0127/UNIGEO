"""Run Qwen VL adapter on cached embeddings."""

import argparse
from pathlib import Path

import torch

from llm_bridge.adapter import QwenVLAdapter, AbsoluteEmbeddingBundle


def load_bundle(cache_path: Path, index: int) -> AbsoluteEmbeddingBundle:
    payload = torch.load(cache_path)
    entries = payload["embeddings"] if isinstance(payload, dict) else payload
    entry = entries[index]

    def _to_tensor(key):
        tensor = entry.get(key)
        if tensor is None:
            return None
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        return tensor.to(torch.float32)

    return AbsoluteEmbeddingBundle(
        z_text=_to_tensor("h_text"),
        z_image=_to_tensor("h_image"),
        z_layout=_to_tensor("h_layout")
    )


def main():
    parser = argparse.ArgumentParser(description="Generate text from cached embeddings via Qwen VL adapter")
    parser.add_argument("--cache", type=str, required=True, help="Path to cached .pt file")
    parser.add_argument("--idx", type=int, default=0, help="Index of sample to use")
    parser.add_argument("--instruction", type=str, default="Summarize the document region.", help="Instruction prompt")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-VL-7B-Instruct", help="Qwen model to use")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    args = parser.parse_args()

    cache_path = Path(args.cache)
    bundle = load_bundle(cache_path, args.idx)

    adapter = QwenVLAdapter(model_name=args.model)
    output = adapter.generate(
        bundle,
        args.instruction,
        max_new_tokens=args.max_new_tokens
    )
    print("=== Adapter Output ===")
    print(output)


if __name__ == "__main__":
    main()

