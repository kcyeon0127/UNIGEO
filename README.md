# UniDoc: Multimodal Document Understanding

PDF 문서를 세 가지 모달리티(Text, Image, Layout)의 embedding space로 변환하는 파이프라인.

## Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PDF (Digital / Scanned)                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                      ┌─────────────────────────────┐
                      │    pdf2image (렌더링)        │
                      └─────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Layout Detection (LayoutParser + EfficientDet/PubLayNet)               │
│  ───────────────────────────────────────────────────────                │
│  • Text, Title, List, Table, Figure 영역 탐지                           │
│  • Confidence score 기반 필터링                                         │
│  • Reading order 정렬                                                   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  OCR (EasyOCR)                                                          │
│  ─────────────────                                                      │
│  • Text/Title/List 영역에서 텍스트 추출                                  │
│  • 다국어 지원 (en, ko, ja, zh, ...)                                    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Tokenization (세 모달리티 Embedding)                                   │
│  ─────────────────────────────────────                                  │
│                                                                         │
│    ┌─────────────┐      ┌─────────────┐      ┌─────────────┐           │
│    │ Text Region │      │Figure/Table │      │ All Regions │           │
│    │  (OCR 결과)  │      │  (cropped)  │      │   (bbox)    │           │
│    └──────┬──────┘      └──────┬──────┘      └──────┬──────┘           │
│           │                    │                    │                   │
│           ▼                    ▼                    ▼                   │
│    ┌─────────────┐      ┌─────────────┐      ┌─────────────┐           │
│    │TextEmbedding│      │VisualEmbed  │      │LayoutEmbed  │           │
│    │ (Qwen2-VL)  │      │ (Qwen2-VL)  │      │(Sinusoidal) │           │
│    └──────┬──────┘      └──────┬──────┘      └──────┬──────┘           │
│           │                    │                    │                   │
│           ▼                    ▼                    ▼                   │
│          hᵀ                   hᴵ                   hᴸ                  │
│       (3584,)              (3584,)              (3584,)                 │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Future: Absolute Semantic Space (Z)                                    │
│  ─────────────────────────────────────                                  │
│       hᵀ ──→ [A_T] ──→ zᵀ ─┐                                           │
│                             │                                           │
│       hᴵ ──→ [A_I] ──→ zᴵ ─┼──→  Z 공간 (공통 의미 공간)               │
│                             │                                           │
│       hᴸ ──→ [A_L] ──→ zᴸ ─┘                                           │
│                                                                         │
│  + Alignment Loss (figure-caption-layout contrastive)                   │
│  + Task Loss (QA, classification, ...)                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
unidoc/
├── detection/                  # Vision 기반 Layout Detection + OCR
│   ├── __init__.py
│   ├── layout_detector.py      # LayoutParser + EfficientDet
│   ├── ocr.py                  # EasyOCR 기반 텍스트 추출
│   └── pipeline.py             # 전체 파이프라인 통합
│
├── tokenization/               # 세 모달리티 Embedding
│   ├── __init__.py
│   ├── text_embedding.py       # hᵀ - Qwen2-VL text encoder
│   ├── visual_embedding.py     # hᴵ - Qwen2-VL vision encoder
│   ├── layout_embedding.py     # hᴸ - Sinusoidal + MLP
│   └── region_tokenizer.py     # 통합 tokenizer
│
├── parsing/                    # (Legacy) Digital PDF 파싱
│   ├── pdf_parser.py
│   ├── text_extractor.py
│   ├── image_extractor.py
│   └── graphic_extractor.py
│
└── grouping/                   # (Legacy) Region 그룹핑
    ├── region_detector.py
    ├── table_detector.py
    ├── figure_detector.py
    ├── paragraph_grouper.py
    └── layout_analyzer.py
```

---

## Quick Start

### Installation

```bash
# PyTorch (CUDA 12.1)
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121

# Dependencies
pip install layoutparser easyocr pdf2image transformers pillow numpy
```

### Basic Usage

```python
from unidoc.detection import DetectionPipeline

# 파이프라인 초기화
pipeline = DetectionPipeline(
    # use_lite 플래그 제거 - 항상 정식 Qwen encoder 사용
    ocr_languages=['en'],       # OCR 언어
    detection_confidence=0.3    # detection threshold
)

# PDF 처리
embeddings = pipeline.process_pdf("document.pdf", pages=[1])

# 결과 확인
for emb in embeddings[0]:
    print(f"[{emb.region_id}] {emb.label.value}")
    print(f"    h_text:   {emb.h_text.shape if emb.has_text else None}")
    print(f"    h_image:  {emb.h_image.shape if emb.has_image else None}")
    print(f"    h_layout: {emb.h_layout.shape}")
    print(f"    text: {emb.text[:50]}...")
```

### Output Example

```
[0] Text
    h_text:   torch.Size([3584])
    h_image:  None
    h_layout: torch.Size([3584])
    text: Yvonne Jaqueline Strzechowski (born 30 July 1982)...

[1] Figure
    h_text:   None
    h_image:  torch.Size([3584])
    h_layout: torch.Size([3584])
    text:

[2] Title
    h_text:   torch.Size([3584])
    h_image:  None
    h_layout: torch.Size([3584])
    text: Early life
```

---

## Components

### 1. Layout Detection (`detection/layout_detector.py`)

LayoutParser + EfficientDet (PubLayNet) 기반 layout detection.

```python
from unidoc.detection import LayoutDetector

detector = LayoutDetector(confidence_threshold=0.3)
regions = detector.detect(page_image)

# 탐지 가능한 타입: Text, Title, List, Table, Figure
```

**PubLayNet Labels:**
| Label | Description |
|-------|-------------|
| Text | 본문 텍스트 |
| Title | 제목 |
| List | 리스트 |
| Table | 표 |
| Figure | 그림/차트 |

### 2. OCR (`detection/ocr.py`)

EasyOCR 기반 텍스트 추출.

```python
from unidoc.detection import OCREngine

ocr = OCREngine(languages=['en', 'ko'], gpu=True)
text = ocr.extract_text(image, bbox)
```

**지원 언어:** 80+ (en, ko, ja, zh, ...)

### 3. Text Embedding (`tokenization/text_embedding.py`)

Qwen2-VL의 텍스트 처리 방식 사용.

```python
from unidoc.tokenization import TextEmbedding

encoder = TextEmbedding(model_name="Qwen/Qwen2-VL-7B-Instruct")
h_text = encoder.encode("Hello, world!")  # (3584,)
```

### 4. Visual Embedding (`tokenization/visual_embedding.py`)

Qwen2-VL의 Vision Encoder 사용.

```python
from unidoc.tokenization import VisualEmbedding

encoder = VisualEmbedding(model_name="Qwen/Qwen2-VL-7B-Instruct")
h_image = encoder.encode(pil_image)  # (3584,)
```

### 5. Layout Embedding (`tokenization/layout_embedding.py`)

Sinusoidal positional encoding + Learnable embedding (Hybrid).

```python
from unidoc.tokenization import LayoutEmbedding

encoder = LayoutEmbedding(hidden_size=3584)
h_layout = encoder(
    bbox=(100, 100, 300, 200),
    page_id=0,
    reading_order=0,
    region_type=0,  # PARAGRAPH
    page_width=612,
    page_height=792
)  # (3584,)
```

**입력 Features:**
| Feature | Type | Description |
|---------|------|-------------|
| bbox | (x1, y1, x2, y2) | Bounding box |
| page_id | int | 페이지 번호 |
| reading_order | int | 읽기 순서 |
| region_type | int | Region 타입 ID |

---

## Region Type별 Embedding

| Region Type | hᵀ (Text) | hᴵ (Image) | hᴸ (Layout) |
|-------------|:---------:|:----------:|:-----------:|
| Text        | ✅        | ❌         | ✅          |
| Title       | ✅        | ❌         | ✅          |
| List        | ✅        | ❌         | ✅          |
| Table       | ✅        | ✅         | ✅          |
| Figure      | ❌        | ✅         | ✅          |

---

## Batch Processing

```python
from unidoc.detection import collate_embeddings

# 여러 region embedding을 배치 텐서로 변환
batch = collate_embeddings(embeddings)

print(batch["h_layout"].shape)  # (num_regions, 3584)
print(batch["h_text"].shape)    # (num_text_regions, 3584)
print(batch["h_image"].shape)   # (num_image_regions, 3584)
print(batch["text_mask"])       # (num_regions,) bool
print(batch["image_mask"])      # (num_regions,) bool
```

---

## Target Model: Qwen2-VL 7B

| Component | Parameter | Value |
|-----------|-----------|-------|
| **LLM (Qwen2)** | hidden_size | 3584 |
| | num_hidden_layers | 28 |
| | num_attention_heads | 28 |
| **Vision Encoder** | embed_dim | 1280 |
| | depth | 32 |
| | num_heads | 16 |
| | output projection | 3584 |

**모든 embedding 차원을 3584로 통일** (Qwen2 hidden size에 맞춤)

---

## Two Pipelines

### Pipeline A: Vision-based (권장)

어떤 PDF든 처리 가능 (digital / scanned).

```python
from unidoc.detection import DetectionPipeline

pipeline = DetectionPipeline()
embeddings = pipeline.process_pdf("any_document.pdf")
```

### Pipeline B: Digital PDF Only (Legacy)

PDF 내부 텍스트 레이어가 있는 경우만 사용 가능.

```python
from unidoc.parsing import PDFVectorParser
from unidoc.grouping import RegionDetector
from unidoc.tokenization import RegionTokenizer

parser = PDFVectorParser()
doc = parser.parse("digital_document.pdf")

detector = RegionDetector()
regions = detector.detect(doc.pages[0])

tokenizer = RegionTokenizer()
embeddings = tokenizer.tokenize(regions, page_image, ...)
```

---

## Dependencies

```
torch>=2.2.0
transformers>=4.40.0
layoutparser>=0.3.4
easyocr>=1.7.0
pdf2image>=1.16.0
Pillow>=10.0.0
numpy>=1.26.0
PyMuPDF>=1.23.0  # for legacy pipeline
```

---

## Datasets

| Dataset | Type | Description |
|---------|------|-------------|
| feta_tab | Wikipedia PDF | 표 이해 |
| slidevqa | 슬라이드 | 슬라이드 VQA |
| spiqa | 논문 | 논문 이미지 QA |
| scigraphvqa | 논문 그래프 | 과학 그래프 VQA |

---

## TODO

- [ ] Absolute Semantic Space (Z) 구현
  - [ ] Modality-specific projectors (A_T, A_I, A_L)
  - [ ] Cross-modal alignment loss
- [ ] Qwen2-VL LLM 연결
- [ ] Training pipeline
- [ ] Evaluation on benchmarks

---

## References

- [Qwen2-VL](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)
- [LayoutParser](https://github.com/Layout-Parser/layout-parser)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [PubLayNet](https://github.com/ibm-aur-nlp/PubLayNet)
