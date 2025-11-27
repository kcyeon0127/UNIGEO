"""Train baseline with orthogonal projectors but without contrastive loss."""

import argparse
from pathlib import Path
from typing import Optional, List

from train import train
from unidoc.model import ManifoldAlignModel


def _collect_pdfs(directory: Optional[str]) -> Optional[List[Path]]:
    if directory is None:
        return None
    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    return sorted(dir_path.glob("**/*.pdf"))


def main():
    parser = argparse.ArgumentParser(description="Train no-contrast baseline")
    parser.add_argument("--data_dir", type=str, required=True, help="Training PDF directory")
    parser.add_argument("--val_dir", type=str, default=None, help="Validation PDF directory")
    parser.add_argument("--z_dim", type=int, default=512, help="Absolute space dimension")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--use_lite", action="store_true", help="Use lite encoders")
    parser.add_argument("--save_path", type=str, default="no_contrast_baseline.pt", help="Checkpoint path")
    parser.add_argument("--num_labels", type=int, default=None, help="Number of labels")
    parser.add_argument("--hidden_dim", type=int, default=3584, help="Anchor embedding size")

    args = parser.parse_args()

    train_paths = _collect_pdfs(args.data_dir)
    val_paths = _collect_pdfs(args.val_dir) if args.val_dir else None

    train(
        train_pdf_paths=train_paths,
        val_pdf_paths=val_paths,
        hidden_dim=args.hidden_dim,
        z_dim=args.z_dim,
        num_labels=args.num_labels,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        lambda_align=0.0,
        lambda_ortho=1e-3,
        device=args.device,
        use_lite=args.use_lite,
        save_path=Path(args.save_path) if args.save_path else None,
        model_cls=ManifoldAlignModel,
    )


if __name__ == "__main__":
    main()

