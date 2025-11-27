# UniDoc Tokenization Module

## Overview

PDF 문서의 Region들을 세 가지 모달리티(Text, Image, Layout)의 embedding space로 변환하는 모듈.

```
PDF → [Parsing] → [Grouping] → List[Region]
                                    ↓
                      ┌─────────────┼─────────────┐
                      ↓             ↓             ↓
                 Text view    Image view    Layout view
                      ↓             ↓             ↓
                 Qwen2-VL      Qwen2-VL       Layout
                 Text Enc     Vision Enc     Encoder
                      ↓             ↓             ↓
                     hᵀ            hᴵ            hᴸ
                 (d=3584)      (d=3584)      (d=3584)
```

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

## Module Structure

```
unidoc/tokenization/
├── __init__.py
├── text_embedding.py      # Region 텍스트 → hᵀ
├── visual_embedding.py    # Region 이미지 → hᴵ
├── layout_embedding.py    # Region 좌표/구조 → hᴸ
└── region_tokenizer.py    # 통합: Region → RegionEmbedding
```

---

## 1. Text Embedding (`text_embedding.py`)

### 목적
Region의 텍스트를 Qwen2-VL의 텍스트 처리 방식과 동일하게 embedding

### 구현 방식
```python
class TextEmbedding:
    """
    Qwen2-VL의 text embedding 방식 사용
    - tokenizer: Qwen2-VL tokenizer
    - embedding: model.model.embed_tokens
    """

    def __init__(self, model_name="Qwen/Qwen2-VL-7B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(model_name)
        self.embed_tokens = self.model.model.embed_tokens  # nn.Embedding

    def encode(self, text: str) -> torch.Tensor:
        """
        Args:
            text: Region의 텍스트
        Returns:
            hᵀ: (d=3584,) 텍스트 embedding
        """
        tokens = self.tokenizer(text, return_tensors="pt")
        embeddings = self.embed_tokens(tokens.input_ids)  # (1, seq_len, 3584)
        # Mean pooling over sequence
        h_T = embeddings.mean(dim=1).squeeze(0)  # (3584,)
        return h_T
```

### 입출력
- **Input**: Region.text (str)
- **Output**: hᵀ ∈ ℝ³⁵⁸⁴

### 처리 대상
- 모든 Region (PARAGRAPH, TITLE, HEADING, TABLE, FIGURE caption 등)

---

## 2. Visual Embedding (`visual_embedding.py`)

### 목적
Figure/Table region의 이미지를 Qwen2-VL의 vision encoder로 embedding

### 구현 방식
```python
class VisualEmbedding:
    """
    Qwen2-VL의 Vision Encoder 사용
    - ViT with 32 layers, embed_dim=1280
    - Output projection to 3584
    """

    def __init__(self, model_name="Qwen/Qwen2-VL-7B-Instruct"):
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(model_name)
        self.visual = self.model.visual  # Qwen2VisionTransformerPretrainedModel
        self.processor = AutoProcessor.from_pretrained(model_name)

    def encode(self, image: PIL.Image) -> torch.Tensor:
        """
        Args:
            image: Region에서 crop한 이미지
        Returns:
            hᴵ: (d=3584,) 이미지 embedding
        """
        # Process image
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs.pixel_values

        # Vision encoder forward
        vision_output = self.visual(pixel_values)  # (1, num_patches, 3584)

        # Mean pooling over patches
        h_I = vision_output.mean(dim=1).squeeze(0)  # (3584,)
        return h_I
```

### 입출력
- **Input**: PIL.Image (PDF 페이지에서 Region bbox로 crop)
- **Output**: hᴵ ∈ ℝ³⁵⁸⁴

### 처리 대상
- **FIGURE** region: 이미지 전체
- **TABLE** region: 표 영역 crop
- Text-only region: **처리 안 함** (hᴵ = None)

### 이미지 획득 방식
```python
# PDF 페이지를 이미지로 렌더링
from pdf2image import convert_from_path
page_image = convert_from_path(pdf_path, dpi=150)[page_num]

# Region bbox로 crop
x0, y0, x1, y1 = region.bbox
# PDF 좌표 → 이미지 좌표 변환 (DPI 고려)
cropped = page_image.crop((x0 * scale, y0 * scale, x1 * scale, y1 * scale))
```

---

## 3. Layout Embedding (`layout_embedding.py`)

### 목적
Region의 위치/구조 정보를 embedding

### 입력 Features
| Feature | Type | Description |
|---------|------|-------------|
| x_center | float | 중심 x좌표 (0~1 normalized) |
| y_center | float | 중심 y좌표 (0~1 normalized) |
| width | float | 너비 (0~1 normalized) |
| height | float | 높이 (0~1 normalized) |
| page_id | int | 페이지 번호 |
| reading_order | int | 읽기 순서 |
| region_type | enum | PARAGRAPH, TABLE, FIGURE 등 |

### 구현 방식 (Option D: Hybrid)
```python
class LayoutEmbedding(nn.Module):
    """
    Hybrid approach:
    - 연속값 (좌표): Sinusoidal positional encoding
    - 이산값 (type, page): Learnable embedding
    """

    def __init__(self, hidden_size=3584, num_region_types=12, max_pages=100):
        super().__init__()

        # 좌표 encoding (x, y, w, h 각각)
        coord_dim = 128  # 각 좌표당 128 dim
        self.coord_encoder = SinusoidalPositionalEncoding(dim=coord_dim)
        # 4개 좌표 * 128 = 512

        # 이산 feature embedding
        self.region_type_embed = nn.Embedding(num_region_types, 256)
        self.page_embed = nn.Embedding(max_pages, 128)
        self.reading_order_embed = nn.Embedding(1000, 128)  # max 1000 regions per doc

        # 통합 projection
        # 512 (coords) + 256 (type) + 128 (page) + 128 (order) = 1024
        self.projection = nn.Sequential(
            nn.Linear(1024, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )

    def forward(self,
                bbox: Tuple[float, float, float, float],
                page_id: int,
                reading_order: int,
                region_type: RegionType) -> torch.Tensor:
        """
        Returns:
            hᴸ: (d=3584,) layout embedding
        """
        # Normalize bbox to [0, 1]
        x0, y0, x1, y1 = bbox
        x_c = (x0 + x1) / 2
        y_c = (y0 + y1) / 2
        w = x1 - x0
        h = y1 - y0

        # Sinusoidal encoding for coordinates
        coord_features = torch.cat([
            self.coord_encoder(x_c),
            self.coord_encoder(y_c),
            self.coord_encoder(w),
            self.coord_encoder(h),
        ], dim=-1)  # (512,)

        # Learnable embeddings for discrete features
        type_emb = self.region_type_embed(region_type)  # (256,)
        page_emb = self.page_embed(page_id)  # (128,)
        order_emb = self.reading_order_embed(reading_order)  # (128,)

        # Concatenate and project
        combined = torch.cat([coord_features, type_emb, page_emb, order_emb], dim=-1)
        h_L = self.projection(combined)  # (3584,)

        return h_L
```

### Alternative: LayoutLM 스타일 (Option A 변형)
```python
class LayoutEmbeddingDiscrete(nn.Module):
    """
    LayoutLMv3 스타일: 좌표를 discretize하여 learnable embedding
    """

    def __init__(self, hidden_size=3584, coord_bins=1024):
        super().__init__()

        # 좌표를 [0, 1023] 범위로 discretize 후 embedding
        self.x_embed = nn.Embedding(coord_bins, hidden_size // 4)
        self.y_embed = nn.Embedding(coord_bins, hidden_size // 4)
        self.w_embed = nn.Embedding(coord_bins, hidden_size // 4)
        self.h_embed = nn.Embedding(coord_bins, hidden_size // 4)

        # region type, page 등
        self.type_embed = nn.Embedding(12, hidden_size // 4)

        self.projection = nn.Linear(hidden_size + hidden_size // 4, hidden_size)

    def forward(self, bbox, region_type, ...):
        # Discretize coordinates to [0, 1023]
        x = int(bbox[0] * 1023)
        y = int(bbox[1] * 1023)
        w = int((bbox[2] - bbox[0]) * 1023)
        h = int((bbox[3] - bbox[1]) * 1023)

        coord_emb = self.x_embed(x) + self.y_embed(y) + self.w_embed(w) + self.h_embed(h)
        type_emb = self.type_embed(region_type)

        h_L = self.projection(torch.cat([coord_emb, type_emb], dim=-1))
        return h_L
```

### 입출력
- **Input**: Region.bbox, page_id, reading_order, region_type
- **Output**: hᴸ ∈ ℝ³⁵⁸⁴

### 처리 대상
- **모든 Region** (위치 정보는 항상 존재)

---

## 4. Region Tokenizer (`region_tokenizer.py`)

### 목적
세 인코더를 통합하여 Region → RegionEmbedding 변환

### 데이터 구조
```python
@dataclass
class RegionEmbedding:
    """Region의 세 모달리티 embedding"""
    region_id: int
    region_type: RegionType

    h_text: Optional[torch.Tensor]    # (3584,) or None
    h_image: Optional[torch.Tensor]   # (3584,) or None
    h_layout: torch.Tensor            # (3584,) - 항상 존재

    # 메타데이터
    has_text: bool
    has_image: bool
```

### 구현
```python
class RegionTokenizer:
    """Region을 세 모달리티 embedding으로 변환"""

    def __init__(self, model_name="Qwen/Qwen2-VL-7B-Instruct"):
        self.text_encoder = TextEmbedding(model_name)
        self.visual_encoder = VisualEmbedding(model_name)
        self.layout_encoder = LayoutEmbedding()

    def tokenize(self,
                 regions: List[Region],
                 page_image: PIL.Image,
                 page_width: float,
                 page_height: float) -> List[RegionEmbedding]:
        """
        Args:
            regions: RegionDetector 출력
            page_image: 렌더링된 PDF 페이지 이미지
            page_width, page_height: 페이지 크기 (정규화용)

        Returns:
            List[RegionEmbedding]: 각 region의 세 모달리티 embedding
        """
        results = []

        for region in regions:
            # 1. Text embedding
            h_text = None
            if region.has_text:
                h_text = self.text_encoder.encode(region.text)

            # 2. Visual embedding (Figure/Table만)
            h_image = None
            if region.region_type in [RegionType.FIGURE, RegionType.TABLE]:
                cropped = self._crop_region(page_image, region.bbox,
                                           page_width, page_height)
                h_image = self.visual_encoder.encode(cropped)

            # 3. Layout embedding (항상)
            normalized_bbox = self._normalize_bbox(region.bbox,
                                                   page_width, page_height)
            h_layout = self.layout_encoder(
                bbox=normalized_bbox,
                page_id=0,  # 단일 페이지인 경우
                reading_order=region.reading_order,
                region_type=region.region_type
            )

            results.append(RegionEmbedding(
                region_id=region.region_id,
                region_type=region.region_type,
                h_text=h_text,
                h_image=h_image,
                h_layout=h_layout,
                has_text=h_text is not None,
                has_image=h_image is not None
            ))

        return results
```

---

## Region Type별 Embedding 구성

| Region Type | hᵀ (Text) | hᴵ (Image) | hᴸ (Layout) |
|-------------|-----------|------------|-------------|
| PARAGRAPH | ✅ | ❌ | ✅ |
| TITLE | ✅ | ❌ | ✅ |
| HEADING | ✅ | ❌ | ✅ |
| TABLE | ✅ (셀 텍스트) | ✅ (표 이미지) | ✅ |
| FIGURE | ✅ (캡션) | ✅ (그림) | ✅ |
| CAPTION | ✅ | ❌ | ✅ |

---

## Usage Example

```python
from unidoc.parsing import PDFVectorParser
from unidoc.grouping import RegionDetector
from unidoc.tokenization import RegionTokenizer
from pdf2image import convert_from_path

# 1. PDF 파싱
parser = PDFVectorParser()
doc = parser.parse("document.pdf")

# 2. Region 그룹핑
detector = RegionDetector()
regions = detector.detect(doc.pages[0])

# 3. 페이지 이미지 렌더링
page_images = convert_from_path("document.pdf", dpi=150)

# 4. Region Tokenization
tokenizer = RegionTokenizer()
embeddings = tokenizer.tokenize(
    regions=regions,
    page_image=page_images[0],
    page_width=doc.pages[0].width,
    page_height=doc.pages[0].height
)

# 결과
for emb in embeddings:
    print(f"Region {emb.region_id} ({emb.region_type})")
    print(f"  - hᵀ: {emb.h_text.shape if emb.h_text is not None else None}")
    print(f"  - hᴵ: {emb.h_image.shape if emb.h_image is not None else None}")
    print(f"  - hᴸ: {emb.h_layout.shape}")
```

---

## Dependencies

```
torch>=2.0.0
transformers>=4.40.0
Pillow>=10.0.0
pdf2image>=1.16.0
```

---

## TODO (Future: Absolute Semantic Space)

현재는 각 모달리티별 개별 space (hᵀ, hᴵ, hᴸ)만 구현.

추후 구현 예정:
- `projectors.py`: 모달리티별 Linear projector (A_T, A_I, A_L)
- `alignment_loss.py`: Cross-modal contrastive loss
- Z 공간으로의 통합 투영
