"""Utility to extract and cache region embeddings per split."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from train import extract_embeddings_from_pdfs
from unidoc.detection import DetectionPipeline


def _read_pdf_paths_from_txt(txt_path: Path) -> List[Path]:
    lines = [line.strip() for line in txt_path.read_text().splitlines()]
    return [Path(line) for line in lines if line]


def _load_splits(split_file: Path, root: Optional[Path] = None) -> Dict[str, List[Path]]:
    root = root or Path('.')
    data = json.loads(split_file.read_text())
    splits = {}
    for split_name, entries in data.items():
        paths = []
        for entry in entries:
            path = Path(entry)
            if not path.is_absolute():
                path = root / path
            paths.append(path)
        splits[split_name] = paths
    return splits


def _collect_from_dir(directory: Path, max_docs: Optional[int] = None) -> List[Path]:
    pdfs = sorted(directory.glob("**/*.pdf"))
    if max_docs is not None:
        pdfs = pdfs[:max_docs]
    return pdfs


def _save_embeddings(
    pdf_paths: List[Path],
    pipeline: DetectionPipeline,
    output_path: Path,
    pages: Optional[List[int]] = None
):
    if not pdf_paths:
        print(f"[WARN] No PDFs provided for {output_path}, skipping.")
        return
    embeddings = extract_embeddings_from_pdfs(
        pdf_paths,
        pipeline,
        pages=pages,
        show_progress=True,
        progress_desc=f"{output_path.stem}"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, output_path)
    print(f"Saved {len(embeddings)} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Cache region embeddings per split")
    parser.add_argument("--pdf_dir", type=str, help="Directory containing PDFs", default=None)
    parser.add_argument("--pdf_list", type=str, help="Text file listing PDFs", default=None)
    parser.add_argument("--split_file", type=str, help="JSON file with split definitions", default=None)
    parser.add_argument("--split_root", type=str, help="Root directory for relative paths in split file", default=None)
    parser.add_argument("--output", type=str, help="Output .pt file (single split)", default=None)
    parser.add_argument("--output_dir", type=str, help="Directory to store split caches", default=None)
    parser.add_argument("--max_docs", type=int, help="Limit number of PDFs processed", default=None)
    parser.add_argument("--pages", type=str, help="Comma-separated list of page numbers (1-indexed)", default=None)
    parser.add_argument("--use_lite", action="store_true", help="Use lite encoders in DetectionPipeline")
    parser.add_argument("--dpi", type=int, default=150, help="PDF rendering DPI")

    args = parser.parse_args()

    if args.split_file:
        if not args.output_dir:
            raise ValueError("--output_dir is required when using --split_file")
    else:
        if not args.output:
            raise ValueError("--output is required when not using --split_file")

    if args.split_file:
        split_root = Path(args.split_root) if args.split_root else Path('.').resolve()
        splits = _load_splits(Path(args.split_file), split_root)
    else:
        splits = {}
        if args.pdf_list:
            paths = _read_pdf_paths_from_txt(Path(args.pdf_list))
        elif args.pdf_dir:
            paths = _collect_from_dir(Path(args.pdf_dir), args.max_docs)
        else:
            raise ValueError("Either --pdf_dir or --pdf_list must be provided")
        splits["cache"] = paths

    pages = [int(p) for p in args.pages.split(',')] if args.pages else None

    pipeline = DetectionPipeline(use_lite=args.use_lite, detection_confidence=0.3)

    if args.split_file:
        output_dir = Path(args.output_dir)
        for split_name, pdf_paths in splits.items():
            if args.max_docs is not None:
                pdf_paths = pdf_paths[:args.max_docs]
            output_path = output_dir / f"{split_name}.pt"
            _save_embeddings(pdf_paths, pipeline, output_path, pages=pages)
    else:
        output_path = Path(args.output)
        _save_embeddings(splits["cache"], pipeline, output_path, pages=pages)


if __name__ == "__main__":
    main()
