"""Augment cached region embeddings with question embeddings."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

from unidoc.tokenization.text_embedding import TextEmbedding


def _load_cache(path: Path) -> Tuple[List[dict], Optional[dict]]:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "embeddings" in payload:
        return payload["embeddings"], payload
    if isinstance(payload, list):
        return payload, None
    raise ValueError(f"Unsupported cache format at {path}")


def _save_cache(path: Path, embeddings: List[dict], payload: Optional[dict]) -> None:
    if payload is None:
        torch.save(embeddings, path)
    else:
        payload["embeddings"] = embeddings
        torch.save(payload, path)


def _load_doc_list(doc_list_path: Optional[Path]) -> Optional[List[str]]:
    if doc_list_path is None:
        return None
    lines: List[str] = []
    for raw in doc_list_path.read_text().splitlines():
        if raw.strip():
            lines.append(raw.strip())
    return lines


def _load_split_docs(
    split_file: Optional[Path],
    split_name: Optional[str],
    split_root: Optional[Path],
) -> Optional[List[str]]:
    if split_file is None or split_name is None:
        return None

    data = json.loads(split_file.read_text())
    if split_name not in data:
        raise ValueError(f"Split '{split_name}' not found in {split_file}")

    docs: List[str] = []
    for entry in data[split_name]:
        path = Path(entry)
        if not path.is_absolute() and split_root:
            path = Path(split_root) / path
        docs.append(path.name)
    return docs


def _normalize_doc_id(value: str) -> str:
    return Path(value).name.lower()


def _load_question_rows(
    qa_path: Path,
    doc_column: str,
    question_column: str,
    split_column: Optional[str],
    splits: Optional[Sequence[str]],
) -> List[Dict[str, str]]:
    if qa_path.suffix.lower() == ".json":
        data = json.loads(qa_path.read_text())
        if isinstance(data, list):
            rows = data
        elif isinstance(data, dict):
            rows = data.get("data", [])
        else:
            raise ValueError("JSON QA file must be a list or dict with 'data' key")
    else:
        with qa_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

    if split_column and splits:
        allowed = set(s.lower() for s in splits)
        rows = [row for row in rows if row.get(split_column, "").lower() in allowed]

    filtered = []
    for row in rows:
        doc_value = row.get(doc_column)
        question_value = row.get(question_column)
        if not doc_value or not question_value:
            continue
        filtered.append(row)
    return filtered


def _build_question_map(
    rows: List[Dict[str, str]],
    doc_column: str,
    question_column: str,
) -> Dict[str, List[Dict[str, str]]]:
    mapping: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        doc_id = _normalize_doc_id(row[doc_column])
        question = row.get(question_column, "")
        if not question:
            continue
        entry = dict(row)
        entry["_question_text"] = question.strip()
        mapping[doc_id].append(entry)
    return mapping


def _detect_existing(embeddings: List[dict]) -> int:
    return sum(1 for sample in embeddings if "question_embedding" in sample)


def _resolve_doc_ids(
    embeddings: List[dict],
    doc_key: str,
    doc_list: Optional[List[str]],
    split_docs: Optional[List[str]],
) -> List[str]:
    resolved: List[str] = []
    for idx, sample in enumerate(embeddings):
        doc_value = sample.get(doc_key)
        if isinstance(doc_value, str) and doc_value:
            resolved.append(doc_value)
            continue
        source_list = doc_list or split_docs
        if source_list is not None:
            if len(source_list) != len(embeddings):
                raise ValueError(
                    "Provided doc list/split does not match cache length. "
                    f"List length={len(source_list)}, cache samples={len(embeddings)}"
                )
            resolved.append(source_list[idx])
            continue
        available_keys = list(sample.keys())
        raise ValueError(
            "Unable to determine doc identifier for sample index "
            f"{idx}. Provide --doc_key or --doc_list. Available keys: {available_keys}"
        )
    return resolved


def _assign_question_entries(
    doc_ids: List[str],
    question_map: Dict[str, List[Dict[str, str]]],
    assignment: str,
) -> Tuple[List[Optional[Dict[str, str]]], List[str]]:
    counters: Dict[str, int] = defaultdict(int)
    missing_docs: List[str] = []
    assigned: List[Optional[Dict[str, str]]] = []

    for doc_id in doc_ids:
        norm_id = _normalize_doc_id(doc_id)
        entries = question_map.get(norm_id)
        if not entries:
            assigned.append(None)
            missing_docs.append(norm_id)
            continue

        if assignment == "first":
            chosen = entries[0]
        elif assignment == "concat":
            merged = dict(entries[0])
            merged["_question_text"] = " \n".join(entry["_question_text"] for entry in entries)
            chosen = merged
        elif assignment == "cycle":
            idx = counters[norm_id]
            counters[norm_id] += 1
            chosen = entries[idx % len(entries)]
        else:
            raise ValueError(f"Unsupported assignment strategy: {assignment}")

        assigned.append(chosen)

    return assigned, missing_docs


def _encode_labels(label_values: List[str]) -> Tuple[List[int], Dict[str, int]]:
    unique_labels = sorted(set(label_values))
    mapping = {label: idx for idx, label in enumerate(unique_labels)}
    encoded = [mapping[label] for label in label_values]
    return encoded, mapping


def _encode_unique_questions(unique_texts: Sequence[str], encoder: TextEmbedding) -> Dict[str, torch.Tensor]:
    cache: Dict[str, torch.Tensor] = {}
    for text in tqdm(unique_texts, desc="Encoding questions"):
        cache[text] = encoder.encode(text).cpu()
    return cache


def augment_cache_with_questions(args: argparse.Namespace) -> None:
    cache_path = Path(args.cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_path}")

    if args.label_column and args.keep_missing:
        raise ValueError("--keep_missing cannot be used when --label_column is set")

    embeddings, payload = _load_cache(cache_path)
    total = len(embeddings)
    if total == 0:
        print(f"[SKIP] {cache_path}: no samples found")
        return

    existing = _detect_existing(embeddings)
    if existing == total and not args.force:
        print(f"[SKIP] {cache_path}: question embeddings already present for all samples")
        return
    if existing > 0 and not args.force:
        print(
            f"[INFO] {cache_path}: {existing}/{total} samples already contain question embeddings. "
            "Use --force to recompute."
        )

    qa_rows = _load_question_rows(
        Path(args.qa_file),
        args.doc_column,
        args.question_column,
        args.split_column,
        args.splits,
    )
    question_map = _build_question_map(qa_rows, args.doc_column, args.question_column)
    if not question_map:
        raise ValueError("Question file did not yield any doc/question pairs")

    doc_list = _load_doc_list(Path(args.doc_list)) if args.doc_list else None
    split_docs = None
    if doc_list is None and args.split_file:
        split_root = Path(args.split_root) if args.split_root else None
        split_docs = _load_split_docs(Path(args.split_file), args.split_name, split_root)
    doc_ids = _resolve_doc_ids(embeddings, args.doc_key, doc_list, split_docs)
    sample_entries, missing_docs = _assign_question_entries(doc_ids, question_map, args.assignment)

    sample_questions: List[Optional[str]] = []
    sample_labels: List[Optional[str]] = []
    for entry in sample_entries:
        if entry is None:
            sample_questions.append(None)
            sample_labels.append(None)
            continue
        sample_questions.append(entry.get("_question_text"))
        if args.label_column:
            raw_label = entry.get(args.label_column)
            if isinstance(raw_label, str):
                raw_label = raw_label.strip()
            sample_labels.append(raw_label if raw_label else None)
        else:
            sample_labels.append(None)

    missing_count = sum(1 for q in sample_questions if q is None)
    if missing_count:
        unique_missing = sorted(set(missing_docs))
        print(
            f"[WARN] {cache_path}: {missing_count} samples missing questions. Missing docs: {unique_missing[:5]}"
            + ("..." if len(unique_missing) > 5 else "")
        )

    unique_texts = sorted({q for q in sample_questions if q is not None})
    if not unique_texts:
        raise ValueError("No question texts available after assignment")

    encoder = TextEmbedding(
        model_name=args.model_name,
        mode=args.text_mode,
        num_layers=args.num_layers,
        pooling=args.pooling,
        device=args.device,
    )

    text_cache = _encode_unique_questions(unique_texts, encoder)

    final_label_values: List[str] = []
    kept_embeddings: List[dict] = []
    updated = 0
    dropped_missing = 0
    dropped_label_missing = 0
    iter_labels = sample_labels if args.label_column else [None] * len(sample_questions)

    for sample, question_text, label_value in tqdm(
        zip(embeddings, sample_questions, iter_labels),
        total=len(embeddings),
        desc="Attaching questions",
    ):
        if question_text is None:
            if args.keep_missing:
                kept_embeddings.append(sample)
            else:
                dropped_missing += 1
            continue
        if args.label_column and label_value is None:
            dropped_label_missing += 1
            continue
        if (not args.force) and ("question_embedding" in sample):
            kept_embeddings.append(sample)
            continue
        sample["question_embedding"] = text_cache[question_text].clone()
        sample["question_text"] = question_text
        if args.label_column:
            sample["label_text"] = label_value
            final_label_values.append(label_value)
        kept_embeddings.append(sample)
        updated += 1

    if args.label_column and dropped_label_missing:
        print(f"[WARN] {cache_path}: dropped {dropped_label_missing} samples with missing labels.")

    final_payload = payload
    if args.label_column:
        if not final_label_values:
            raise ValueError("No labels available after processing. Check --label_column")
        label_ids, label_vocab = _encode_labels(final_label_values)
        if final_payload is None:
            final_payload = {}
        final_payload["labels"] = label_ids
        final_payload["label_vocab"] = label_vocab

    output_path: Path
    if args.inplace:
        output_path = cache_path
    elif args.output_path:
        output_path = Path(args.output_path)
    else:
        suffix = args.output_suffix or "_with_q"
        output_path = cache_path.with_name(cache_path.stem + suffix + cache_path.suffix)

    if not args.inplace and output_path != cache_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    _save_cache(output_path, kept_embeddings, final_payload)

    print(
        f"[DONE] {cache_path} -> {output_path}: updated {updated} / {total} samples, "
        f"dropped {dropped_missing if not args.keep_missing else 0} missing, "
        f"dropped_labels {dropped_label_missing if args.label_column else 0}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attach question embeddings to cached samples")
    parser.add_argument("--cache_path", required=True, help="Path to cached embeddings (.pt)")
    parser.add_argument("--qa_file", required=True, help="CSV/JSON file with questions")
    parser.add_argument("--doc_column", default="doc_path", help="Column in QA file for document path")
    parser.add_argument("--question_column", default="question", help="Column with question text")
    parser.add_argument("--label_column", default=None, help="Column with label/answer text")
    parser.add_argument("--split_column", default=None, help="Optional column for split filtering")
    parser.add_argument("--splits", nargs="*", default=None, help="Splits to keep (e.g., train val)")
    parser.add_argument("--doc_key", default="doc_path", help="Key inside cache samples that stores doc id")
    parser.add_argument(
        "--doc_list",
        default=None,
        help="Optional text file listing doc ids per sample when cache lacks doc metadata",
    )
    parser.add_argument("--split_file", default=None, help="Split JSON (like preprocess/cache_embeddings.py)")
    parser.add_argument("--split_name", default=None, help="Split name inside split JSON (train/val/test)")
    parser.add_argument("--split_root", default=None, help="Root directory for relative splits")
    parser.add_argument(
        "--assignment",
        choices=["first", "cycle", "concat"],
        default="cycle",
        help="How to map multiple questions per document",
    )
    parser.add_argument("--model_name", default="Qwen/Qwen2-VL-7B-Instruct", help="Text encoder model")
    parser.add_argument("--text_mode", default="token_only", choices=["token_only", "shallow", "deep"])
    parser.add_argument("--num_layers", type=int, default=4, help="Layers for shallow mode")
    parser.add_argument("--pooling", default="mean", choices=["mean", "cls", "last"], help="Pooling strategy")
    parser.add_argument("--device", default=None, help="Device override (e.g., cuda:0)")
    parser.add_argument("--force", action="store_true", help="Recompute even if embeddings already exist")
    parser.add_argument("--inplace", action="store_true", help="Overwrite the input cache in-place")
    parser.add_argument("--output_path", default=None, help="Explicit path for the augmented cache")
    parser.add_argument("--output_suffix", default="_with_q", help="Suffix when writing alongside input")
    parser.add_argument("--keep_missing", action="store_true", help="Keep samples without matching question")
    return parser.parse_args()


if __name__ == "__main__":
    augment_cache_with_questions(parse_args())
