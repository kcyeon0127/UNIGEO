"""Create train/val/test splits for PDF paths."""

import argparse
import json
import random
from pathlib import Path
from typing import List, Optional


def _collect_pdfs(directory: Path) -> List[Path]:
    return sorted(directory.glob("**/*.pdf"))


def _read_pdf_list(list_path: Path) -> List[Path]:
    lines = [line.strip() for line in list_path.read_text().splitlines()]
    return [Path(line) for line in lines if line]


def main():
    parser = argparse.ArgumentParser(description="Create 60/20/20 PDF splits")
    parser.add_argument("--data_dir", type=str, help="Directory containing PDFs", default=None)
    parser.add_argument("--pdf_list", type=str, help="Text file listing PDFs", default=None)
    parser.add_argument("--output", type=str, required=True, help="Output JSON path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--train_ratio", type=float, default=0.6, help="Train ratio")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Val ratio")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="Test ratio")
    parser.add_argument("--relative_root", type=str, default=None, help="Root to express relative paths")

    args = parser.parse_args()

    if not args.data_dir and not args.pdf_list:
        raise ValueError("Provide --data_dir or --pdf_list")

    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")

    if args.pdf_list:
        pdfs = _read_pdf_list(Path(args.pdf_list))
    else:
        pdfs = _collect_pdfs(Path(args.data_dir))

    if not pdfs:
        raise ValueError("No PDFs found")

    random.seed(args.seed)
    random.shuffle(pdfs)

    n = len(pdfs)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    n_test = n - n_train - n_val

    splits = {
        "train": pdfs[:n_train],
        "val": pdfs[n_train:n_train + n_val],
        "test": pdfs[n_train + n_val: n_train + n_val + n_test]
    }

    root = Path(args.relative_root).resolve() if args.relative_root else None

    serializable = {}
    for split_name, paths in splits.items():
        entries = []
        for path in paths:
            full = path if path.is_absolute() else path.resolve()
            if root is not None:
                try:
                    relative = full.relative_to(root)
                    entries.append(str(relative))
                except ValueError:
                    entries.append(str(full))
            else:
                entries.append(str(full))
        serializable[split_name] = entries

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(serializable, indent=2))
    print(f"Saved split file with {n} PDFs to {output_path}")


if __name__ == "__main__":
    main()

