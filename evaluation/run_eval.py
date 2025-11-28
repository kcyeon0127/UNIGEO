"""Evaluate Qwen adapter outputs against reference answers."""

import argparse
import json
from pathlib import Path

from tqdm import tqdm
import torch

from llm_bridge.adapter import QwenVLAdapter, AbsoluteEmbeddingBundle


def load_cache(cache_path: Path):
    payload = torch.load(cache_path)
    if isinstance(payload, dict) and "embeddings" in payload:
        embeddings = payload["embeddings"]
        labels = payload.get("labels")
    else:
        embeddings = payload
        labels = None
    return embeddings, labels


def load_bundle(entry) -> AbsoluteEmbeddingBundle:
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


def exact_match(pred: str, ref: str) -> float:
    return 1.0 if pred.strip().lower() == ref.strip().lower() else 0.0


def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen adapter outputs against references")
    parser.add_argument("--cache", type=str, required=True, help="Cached embedding .pt file")
    parser.add_argument("--references", type=str, required=True, help="JSON list of reference answers")
    parser.add_argument("--instruction", type=str, default="Answer the question using the region.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    args = parser.parse_args()

    cache_path = Path(args.cache)
    references = json.loads(Path(args.references).read_text())

    embeddings, _ = load_cache(cache_path)
    adapter = QwenVLAdapter(model_name=args.model)

    em_scores = []
    for entry, ref in tqdm(list(zip(embeddings, references)), desc="Eval"):
        bundle = load_bundle(entry)
        output = adapter.generate(
            bundle,
            args.instruction,
            max_new_tokens=args.max_new_tokens
        )
        em_scores.append(exact_match(output, ref))

    print(f"Exact Match: {sum(em_scores)/len(em_scores):.4f}")


if __name__ == "__main__":
    main()

