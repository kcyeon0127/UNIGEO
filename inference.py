import os
import json
import argparse
from pathlib import Path
import pandas as pd
import torch
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from pdf2image import convert_from_path
from PIL import Image
import tempfile
"""
Qwen-VL 모델로 성능을 평가하기 위한 코드

사용법 예시:
python inference.py --dataset paper_tab --gpu 3 --num_gpus 2 --use_all_pages

"""

def load_model(model_name="Qwen/Qwen2-VL-7B-Instruct", device="cuda", max_memory=None):
    """Load Qwen2-VL model and processor."""
    print(f"Loading model: {model_name}")

    # Configure device map to use multiple GPUs
    if max_memory:
        print(f"Using multi-GPU with max_memory: {max_memory}")
        device_map_config = {
            "device_map": "auto",
            "max_memory": max_memory
        }
    else:
        device_map_config = {
            "device_map": "auto"
        }

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        **device_map_config
    )

    processor = AutoProcessor.from_pretrained(model_name)

    # Print actual device placement
    if hasattr(model, 'hf_device_map'):
        print("Model device map:")
        for name, device in model.hf_device_map.items():
            print(f"  {name}: {device}")

    print(f"Model loaded successfully")
    return model, processor


def pdf_to_images(pdf_path, dpi=72):
    """Convert PDF to list of PIL images.

    Args:
        pdf_path: Path to PDF file
        dpi: DPI for image conversion (72 for memory efficiency)
    """
    try:
        images = convert_from_path(pdf_path, dpi=dpi)
        return images
    except Exception as e:
        print(f"Error converting PDF {pdf_path}: {e}")
        raise


def inference_single(model, processor, pdf_path, question, max_new_tokens=512, use_first_page_only=True):
    """Run inference on a single PDF-question pair."""

    # Convert PDF to images
    images = pdf_to_images(pdf_path)

    if use_first_page_only:
        # Use only the first page
        selected_images = [images[0]]
        # Free memory from unused pages
        del images
    else:
        selected_images = images

    # Create messages with images
    content = []
    for img in selected_images:
        # Save to temporary file for qwen_vl_utils
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            img.save(tmp.name, 'PNG')
            content.append({
                "type": "image",
                "image": tmp.name,
            })

    # Free PIL images from memory
    del selected_images

    content.append({"type": "text", "text": question})

    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    # Clean up temporary files
    for item in content:
        if item.get("type") == "image":
            try:
                os.unlink(item["image"])
            except:
                pass

    # Explicitly free GPU memory
    del inputs
    del generated_ids
    del generated_ids_trimmed
    del image_inputs
    del video_inputs
    torch.cuda.empty_cache()

    return output_text[0]


def process_dataset(dataset_name, model, processor, output_dir, limit=None, use_all_pages=False):
    """Process entire dataset and generate responses."""

    # Read CSV file
    csv_path = f"data/{dataset_name}/{dataset_name}.csv"
    print(f"Reading dataset from {csv_path}")
    df = pd.read_csv(csv_path)

    if limit:
        df = df.head(limit)
        print(f"Processing first {limit} samples")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing {len(df)} samples...")

    # Statistics
    success_count = 0
    error_count = 0

    # Use tqdm for progress bar
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing samples"):
        q_id = row['q_id']
        doc_id = row['doc_id']
        question = row['question']
        gt_answer = row['answer']
        doc_path = row['doc_path']

        # Full path to document
        full_doc_path = f"data/{dataset_name}/docs/{doc_path}"

        if not os.path.exists(full_doc_path):
            tqdm.write(f"Warning: Document not found: {full_doc_path}")
            error_count += 1
            continue

        try:
            # Run inference
            answer = inference_single(model, processor, full_doc_path, question,
                                    use_first_page_only=not use_all_pages)

            # Save result
            result = {
                "q_id": q_id,
                "doc_id": doc_id,
                "question": question,
                "Answer": answer,
                "gt_answer": gt_answer,
                "doc_path": doc_path
            }

            output_path = os.path.join(output_dir, f"{q_id}.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            success_count += 1
            tqdm.write(f"✓ [{success_count}/{len(df)}] q_id: {q_id} | Answer: {answer[:80]}...")

        except Exception as e:
            error_count += 1
            tqdm.write(f"✗ Error processing q_id {q_id}: {e}")
            continue

    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Success: {success_count}/{len(df)} samples")
    print(f"Errors: {error_count}/{len(df)} samples")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Qwen2-VL inference for UniDoc evaluation")
    parser.add_argument("--dataset", type=str, required=True,
                       choices=["feta_tab", "paper_tab", "scigraphvqa", "slidevqa", "spiqa"],
                       help="Dataset name")
    parser.add_argument("--output_dir", type=str, default="responses",
                       help="Output directory for responses")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-VL-7B-Instruct",
                       help="Model name or path")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of samples to process (for testing)")
    parser.add_argument("--gpu", type=int, default=0,
                       help="GPU device ID")
    parser.add_argument("--use_all_pages", action="store_true",
                       help="Use all pages of PDF instead of just the first page")
    parser.add_argument("--num_gpus", type=int, default=1,
                       help="Number of GPUs to use (for model parallelism)")

    args = parser.parse_args()

    # Set visible GPUs
    if args.num_gpus > 1:
        gpu_ids = ",".join([str(args.gpu + i) for i in range(args.num_gpus)])
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
        print(f"Using GPUs: {gpu_ids}")
        # Configure max memory per GPU
        max_memory = {i: "16GB" for i in range(args.num_gpus)}
        max_memory["cpu"] = "30GB"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        max_memory = None

    # Create output directory
    output_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model, processor = load_model(args.model_name, max_memory=max_memory)

    # Process dataset
    process_dataset(args.dataset, model, processor, output_dir, args.limit, args.use_all_pages)

    print(f"\nDone! Results saved to {output_dir}")
    print(f"Run evaluation with: python eval.py {output_dir}")


if __name__ == "__main__":
    main()
