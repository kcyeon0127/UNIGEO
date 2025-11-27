# UniDoc: Vector-Native Unified Region Tokenization 구현 계획

## 핵심 아이디어 요약

> PDF vector layer에서 직접 text, image, layout 정보를 추출하여
> **처음부터 단일 token modality로 통합** (fusion이 아닌 분해 자체를 제거)

---

## 프로젝트 구조 (최종)

```
/workspace/UniDoc/
├── unidoc/                          # 메인 패키지
│   ├── __init__.py
│   │
│   ├── parsing/                     # Step 1: PDF Vector Parsing
│   │   ├── __init__.py
│   │   ├── pdf_parser.py           # PDF vector layer 추출 (PyMuPDF)
│   │   ├── text_extractor.py       # 텍스트 runs, font, size 추출
│   │   ├── graphic_extractor.py    # drawing paths, lines, rectangles
│   │   └── image_extractor.py      # embedded images 추출
│   │
│   ├── grouping/                    # Step 2: Region Grouping
│   │   ├── __init__.py
│   │   ├── region_detector.py      # 통합 region detection
│   │   ├── paragraph_grouper.py    # 문단 grouping
│   │   ├── table_detector.py       # 표 detection (lines + alignment)
│   │   ├── figure_detector.py      # figure/image + caption association
│   │   └── layout_analyzer.py      # header/footer, reading order
│   │
│   ├── tokenization/                # Step 3: Unified Region Tokenization ⭐
│   │   ├── __init__.py
│   │   ├── region_tokenizer.py     # 핵심: unified token 생성
│   │   ├── text_embedding.py       # text embedding module
│   │   ├── visual_embedding.py     # visual patch embedding
│   │   ├── layout_embedding.py     # 2D layout embedding
│   │   └── role_embedding.py       # region-role embedding
│   │
│   ├── model/                       # Step 4: Layout-Aware Transformer
│   │   ├── __init__.py
│   │   ├── unidoc_model.py         # 메인 모델 아키텍처
│   │   ├── layout_attention.py     # layout-aware attention bias
│   │   ├── region_encoder.py       # region token encoder
│   │   └── llm_adapter.py          # LLM 연결 (Qwen adapter)
│   │
│   ├── data/                        # 데이터 처리
│   │   ├── __init__.py
│   │   ├── dataset.py              # PyTorch Dataset
│   │   ├── collator.py             # Batch collation
│   │   └── preprocessor.py         # 전처리 파이프라인
│   │
│   └── utils/                       # 유틸리티
│       ├── __init__.py
│       ├── bbox_utils.py           # bounding box 연산
│       ├── coord_utils.py          # 좌표 변환
│       └── visualization.py        # 디버깅용 시각화
│
├── configs/                         # 설정 파일
│   ├── model_config.yaml
│   ├── training_config.yaml
│   └── data_config.yaml
│
├── scripts/                         # 실행 스크립트
│   ├── train.py
│   ├── inference_unidoc.py         # 새로운 inference
│   └── preprocess_pdfs.py          # PDF 전처리
│
├── data/                            # 기존 데이터셋 (유지)
├── responses/                       # 출력 디렉토리 (유지)
├── eval.py                          # 기존 평가 스크립트 (유지)
├── inference.py                     # 기존 baseline (유지)
└── requirements.txt                 # 의존성 추가
```

---

## 구현 단계별 상세 계획

### Phase 1: PDF Vector Parsing (Week 1)

#### 1.1 `parsing/pdf_parser.py`
```python
class PDFVectorParser:
    """PDF vector layer에서 모든 요소 추출"""

    def parse(self, pdf_path: str) -> List[PageContent]:
        """
        Returns:
            List[PageContent]: 페이지별 vector 요소들
                - text_blocks: List[TextBlock]
                - drawing_paths: List[DrawingPath]
                - images: List[EmbeddedImage]
                - page_size: Tuple[float, float]
        """
```

**추출할 정보:**
- Text runs: characters, font_name, font_size, color, bbox
- Drawing paths: lines, rectangles, curves (표 구조 파악용)
- Embedded images: image data, bbox, resolution
- Page metadata: size, rotation, crop box

**라이브러리:** PyMuPDF (fitz) - vector layer 직접 접근 가능

#### 1.2 `parsing/text_extractor.py`
```python
class TextExtractor:
    def extract_text_blocks(self, page) -> List[TextBlock]:
        """
        TextBlock:
            - text: str
            - bbox: Tuple[x0, y0, x1, y1]
            - font_name: str
            - font_size: float
            - font_flags: int (bold, italic 등)
            - color: Tuple[r, g, b]
            - char_positions: List[CharPosition]  # 글자별 위치
        """
```

#### 1.3 `parsing/graphic_extractor.py`
```python
class GraphicExtractor:
    def extract_drawings(self, page) -> List[DrawingPath]:
        """
        DrawingPath:
            - path_type: 'line' | 'rect' | 'curve'
            - points: List[Tuple[x, y]]
            - stroke_color: Optional[Tuple]
            - fill_color: Optional[Tuple]
            - line_width: float
        """
```
→ 표의 선(border), 구분선, 그래프 요소 파악

#### 1.4 `parsing/image_extractor.py`
```python
class ImageExtractor:
    def extract_images(self, page) -> List[EmbeddedImage]:
        """
        EmbeddedImage:
            - image_data: bytes (PNG/JPEG)
            - bbox: Tuple[x0, y0, x1, y1]
            - width, height: int
            - colorspace: str
        """
```

---

### Phase 2: Region Grouping (Week 2)

#### 2.1 `grouping/region_detector.py`
```python
class RegionDetector:
    """통합 region detection 파이프라인"""

    def detect_regions(self, page_content: PageContent) -> List[Region]:
        """
        Region:
            - region_id: str
            - region_type: 'paragraph' | 'table' | 'figure' | 'header' | 'footer' | 'caption'
            - bbox: Tuple[x0, y0, x1, y1]
            - content: Union[TextBlock, TableStructure, ImageContent]
            - children: List[Region]  # nested regions (table cells 등)
            - reading_order: int
        """
```

#### 2.2 `grouping/table_detector.py`
```python
class TableDetector:
    """PDF drawing paths 기반 표 detection"""

    def detect_tables(self, text_blocks, drawing_paths) -> List[TableStructure]:
        """
        TableStructure:
            - bbox: Tuple
            - rows: int
            - cols: int
            - cells: List[TableCell]
                - cell_bbox: Tuple
                - row_span, col_span: int
                - content: List[TextBlock]
            - has_header: bool
        """

    def _find_grid_lines(self, drawing_paths) -> GridLines:
        """수평/수직 선 추출하여 그리드 구성"""

    def _align_text_to_cells(self, text_blocks, grid) -> List[TableCell]:
        """텍스트를 셀에 할당"""
```

**핵심 알고리즘:**
1. Drawing paths에서 수평/수직 선 추출
2. 선들의 교차점으로 그리드 구성
3. 텍스트 블록을 셀에 매핑
4. 병합 셀 탐지 (span 계산)

#### 2.3 `grouping/paragraph_grouper.py`
```python
class ParagraphGrouper:
    """텍스트 블록들을 문단으로 grouping"""

    def group_paragraphs(self, text_blocks: List[TextBlock]) -> List[Paragraph]:
        """
        Grouping 기준:
            - 수직 거리 (line spacing)
            - 수평 정렬 (indentation)
            - 폰트 일관성
            - 줄바꿈 패턴
        """
```

#### 2.4 `grouping/figure_detector.py`
```python
class FigureDetector:
    """이미지 + 캡션 연결"""

    def detect_figures(self, images, text_blocks) -> List[FigureRegion]:
        """
        FigureRegion:
            - image: EmbeddedImage
            - caption: Optional[TextBlock]
            - figure_number: Optional[str]
        """

    def _find_caption(self, image_bbox, text_blocks) -> Optional[TextBlock]:
        """이미지 근처에서 'Figure X', 'Fig.' 등 패턴 탐색"""
```

#### 2.5 `grouping/layout_analyzer.py`
```python
class LayoutAnalyzer:
    """전체 페이지 레이아웃 분석"""

    def analyze(self, regions: List[Region]) -> LayoutStructure:
        """
        - header/footer 탐지
        - 다단(multi-column) 구조 파악
        - reading order 결정
        - region 계층 구조화
        """

    def _detect_columns(self, regions) -> int:
        """페이지의 컬럼 수 추정"""

    def _compute_reading_order(self, regions) -> List[int]:
        """좌→우, 위→아래 + 컬럼 고려한 reading order"""
```

---

### Phase 3: Unified Region Tokenization ⭐ (Week 3-4) - 핵심

#### 3.1 `tokenization/region_tokenizer.py`
```python
class UnifiedRegionTokenizer:
    """
    핵심 모듈: text, image, layout을 단일 token modality로 통합

    기존 방식:
        text_tokens = text_encoder(text)
        image_tokens = vit(image)
        layout_embed = layout_encoder(bbox)
        → fusion later

    우리 방식:
        unified_token = concat(text_emb, visual_emb, layout_emb, role_emb)
        → 처음부터 하나의 modality
    """

    def __init__(self, config: TokenizerConfig):
        self.text_embedder = TextEmbedding(config)
        self.visual_embedder = VisualEmbedding(config)
        self.layout_embedder = LayoutEmbedding(config)
        self.role_embedder = RoleEmbedding(config)

        self.fusion_layer = nn.Linear(
            config.text_dim + config.visual_dim + config.layout_dim + config.role_dim,
            config.hidden_dim
        )

    def tokenize(self, regions: List[Region], page_image: Image) -> RegionTokens:
        """
        Args:
            regions: Region Grouping 결과
            page_image: 페이지 렌더링 이미지 (visual patch용)

        Returns:
            RegionTokens:
                - tokens: Tensor[num_regions, hidden_dim]
                - attention_mask: Tensor[num_regions]
                - region_types: List[str]
                - bboxes: Tensor[num_regions, 4]
        """
        unified_tokens = []

        for region in regions:
            # 1. Text embedding (있는 경우)
            text_emb = self.text_embedder(region.text_content)

            # 2. Visual embedding (region crop의 patch)
            region_crop = self._crop_region(page_image, region.bbox)
            visual_emb = self.visual_embedder(region_crop)

            # 3. Layout embedding (정규화된 2D 좌표)
            layout_emb = self.layout_embedder(region.bbox, page_size)

            # 4. Role embedding (region type)
            role_emb = self.role_embedder(region.region_type)

            # 5. 통합 ⭐
            unified = torch.cat([text_emb, visual_emb, layout_emb, role_emb], dim=-1)
            unified = self.fusion_layer(unified)

            unified_tokens.append(unified)

        return RegionTokens(
            tokens=torch.stack(unified_tokens),
            bboxes=torch.tensor([r.bbox for r in regions]),
            region_types=[r.region_type for r in regions]
        )
```

#### 3.2 `tokenization/text_embedding.py`
```python
class TextEmbedding(nn.Module):
    """Region 내 텍스트를 embedding"""

    def __init__(self, config):
        self.tokenizer = AutoTokenizer.from_pretrained(config.text_model)
        self.encoder = AutoModel.from_pretrained(config.text_model)
        # 또는 learnable embedding layer

        self.projector = nn.Linear(self.encoder.config.hidden_size, config.text_dim)

    def forward(self, text: str) -> Tensor:
        """
        텍스트가 없으면 zero vector 반환
        텍스트가 있으면 [CLS] token 또는 mean pooling
        """
        if not text or text.strip() == "":
            return torch.zeros(self.config.text_dim)

        tokens = self.tokenizer(text, return_tensors='pt', truncation=True)
        outputs = self.encoder(**tokens)
        pooled = outputs.last_hidden_state[:, 0, :]  # [CLS]
        return self.projector(pooled.squeeze(0))
```

#### 3.3 `tokenization/visual_embedding.py`
```python
class VisualEmbedding(nn.Module):
    """Region crop을 visual embedding으로"""

    def __init__(self, config):
        self.patch_size = config.patch_size  # e.g., 16
        self.vit = ViTModel.from_pretrained(config.vit_model)
        # 또는 경량화된 CNN

        self.projector = nn.Linear(self.vit.config.hidden_size, config.visual_dim)

    def forward(self, region_image: Image) -> Tensor:
        """
        Region crop 이미지 → visual embedding

        작은 region: resize + pad
        큰 region: adaptive pooling
        """
        # Resize to fixed size
        image = self.transform(region_image)

        # ViT forward
        outputs = self.vit(image.unsqueeze(0))
        pooled = outputs.last_hidden_state[:, 0, :]  # [CLS]

        return self.projector(pooled.squeeze(0))
```

#### 3.4 `tokenization/layout_embedding.py`
```python
class LayoutEmbedding(nn.Module):
    """2D spatial layout embedding"""

    def __init__(self, config):
        self.x_embedding = nn.Embedding(config.max_position, config.layout_dim // 4)
        self.y_embedding = nn.Embedding(config.max_position, config.layout_dim // 4)
        self.w_embedding = nn.Embedding(config.max_position, config.layout_dim // 4)
        self.h_embedding = nn.Embedding(config.max_position, config.layout_dim // 4)

        # 또는 continuous embedding with MLP
        self.mlp = nn.Sequential(
            nn.Linear(4, config.layout_dim),
            nn.ReLU(),
            nn.Linear(config.layout_dim, config.layout_dim)
        )

    def forward(self, bbox: Tuple, page_size: Tuple) -> Tensor:
        """
        bbox: (x0, y0, x1, y1)
        page_size: (width, height)

        정규화: [0, 1] 범위로
        """
        x0, y0, x1, y1 = bbox
        w, h = page_size

        # Normalize to [0, 1]
        normalized = torch.tensor([
            x0 / w, y0 / h,
            (x1 - x0) / w,  # width
            (y1 - y0) / h   # height
        ])

        return self.mlp(normalized)
```

#### 3.5 `tokenization/role_embedding.py`
```python
class RoleEmbedding(nn.Module):
    """Region type/role embedding"""

    REGION_TYPES = [
        'paragraph', 'title', 'header', 'footer',
        'table', 'table_cell', 'table_header',
        'figure', 'caption', 'list', 'equation',
        'page_number', 'footnote', 'unknown'
    ]

    def __init__(self, config):
        self.embedding = nn.Embedding(len(self.REGION_TYPES), config.role_dim)
        self.type_to_idx = {t: i for i, t in enumerate(self.REGION_TYPES)}

    def forward(self, region_type: str) -> Tensor:
        idx = self.type_to_idx.get(region_type, self.type_to_idx['unknown'])
        return self.embedding(torch.tensor(idx))
```

---

### Phase 4: Layout-Aware Transformer (Week 5)

#### 4.1 `model/layout_attention.py`
```python
class LayoutAwareAttention(nn.Module):
    """
    Spatial proximity 기반 attention bias

    기본 attention: softmax(QK^T / sqrt(d))
    우리 방식: softmax((QK^T + spatial_bias) / sqrt(d))
    """

    def __init__(self, config):
        self.num_heads = config.num_heads
        self.spatial_bias_layer = nn.Linear(4, config.num_heads)  # relative position → bias

    def compute_spatial_bias(self, bboxes: Tensor) -> Tensor:
        """
        Args:
            bboxes: [batch, num_regions, 4]

        Returns:
            spatial_bias: [batch, num_heads, num_regions, num_regions]
        """
        # Pairwise relative positions
        # bbox_i와 bbox_j 사이의 관계:
        #   - 수평 거리
        #   - 수직 거리
        #   - overlap 여부
        #   - 포함 관계

        batch, n, _ = bboxes.shape

        # Center points
        cx = (bboxes[:, :, 0] + bboxes[:, :, 2]) / 2
        cy = (bboxes[:, :, 1] + bboxes[:, :, 3]) / 2

        # Pairwise distances
        dx = cx.unsqueeze(2) - cx.unsqueeze(1)  # [batch, n, n]
        dy = cy.unsqueeze(2) - cy.unsqueeze(1)

        # Relative features
        rel_features = torch.stack([dx, dy, dx.abs(), dy.abs()], dim=-1)

        # Convert to attention bias
        bias = self.spatial_bias_layer(rel_features)  # [batch, n, n, num_heads]
        return bias.permute(0, 3, 1, 2)  # [batch, num_heads, n, n]

    def forward(self, Q, K, V, bboxes):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
        spatial_bias = self.compute_spatial_bias(bboxes)
        attn_scores = attn_scores + spatial_bias
        attn_probs = F.softmax(attn_scores, dim=-1)
        return torch.matmul(attn_probs, V)
```

#### 4.2 `model/region_encoder.py`
```python
class RegionEncoder(nn.Module):
    """Region tokens를 처리하는 Transformer encoder"""

    def __init__(self, config):
        self.layers = nn.ModuleList([
            LayoutAwareTransformerLayer(config)
            for _ in range(config.num_layers)
        ])
        self.norm = nn.LayerNorm(config.hidden_dim)

    def forward(self, region_tokens: RegionTokens) -> Tensor:
        """
        Args:
            region_tokens: UnifiedRegionTokenizer의 출력

        Returns:
            encoded: [batch, num_regions, hidden_dim]
        """
        x = region_tokens.tokens
        bboxes = region_tokens.bboxes

        for layer in self.layers:
            x = layer(x, bboxes)

        return self.norm(x)
```

#### 4.3 `model/unidoc_model.py`
```python
class UniDocModel(nn.Module):
    """전체 모델 아키텍처"""

    def __init__(self, config):
        self.pdf_parser = PDFVectorParser()
        self.region_detector = RegionDetector(config)
        self.tokenizer = UnifiedRegionTokenizer(config)
        self.encoder = RegionEncoder(config)
        self.llm_adapter = LLMAdapter(config)

    def forward(self, pdf_path: str, question: str) -> str:
        # Step 1: PDF Parsing
        page_contents = self.pdf_parser.parse(pdf_path)

        all_region_tokens = []
        for page_idx, page_content in enumerate(page_contents):
            # Step 2: Region Grouping
            regions = self.region_detector.detect_regions(page_content)

            # Step 3: Unified Tokenization
            page_image = self._render_page(pdf_path, page_idx)
            region_tokens = self.tokenizer.tokenize(regions, page_image)
            all_region_tokens.append(region_tokens)

        # Step 4: Encoding
        combined_tokens = self._combine_pages(all_region_tokens)
        encoded = self.encoder(combined_tokens)

        # Step 5: LLM Generation
        answer = self.llm_adapter.generate(encoded, question)
        return answer
```

#### 4.4 `model/llm_adapter.py`
```python
class LLMAdapter(nn.Module):
    """Region encoding을 LLM에 연결"""

    def __init__(self, config):
        self.llm = AutoModelForCausalLM.from_pretrained(config.llm_model)
        self.tokenizer = AutoTokenizer.from_pretrained(config.llm_model)

        # Region encoding → LLM embedding space projection
        self.projector = nn.Linear(config.hidden_dim, self.llm.config.hidden_size)

    def generate(self, region_encoding: Tensor, question: str) -> str:
        """
        region_encoding: [num_regions, hidden_dim]
        question: 질문 텍스트

        Returns:
            answer: 생성된 답변
        """
        # Project to LLM space
        document_embedding = self.projector(region_encoding)  # [num_regions, llm_dim]

        # Pool or keep as sequence
        doc_tokens = document_embedding  # region별 토큰으로 전달

        # Prepare question
        question_tokens = self.tokenizer(question, return_tensors='pt')
        question_embeds = self.llm.get_input_embeddings()(question_tokens.input_ids)

        # Combine: [DOC_TOKENS] + [SEP] + [QUESTION]
        inputs_embeds = torch.cat([doc_tokens, question_embeds], dim=1)

        # Generate
        outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=256,
            do_sample=False
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
```

---

### Phase 5: Training & Evaluation (Week 6)

#### 5.1 `data/dataset.py`
```python
class UniDocDataset(Dataset):
    """PDF VQA 데이터셋"""

    def __init__(self, csv_path: str, config):
        self.data = pd.read_csv(csv_path)
        self.pdf_parser = PDFVectorParser()
        self.region_detector = RegionDetector(config)
        self.config = config

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Parse PDF (캐싱 가능)
        pdf_path = row['doc_path']
        page_contents = self.pdf_parser.parse(pdf_path)

        # Region detection
        all_regions = []
        page_images = []
        for page_idx, page_content in enumerate(page_contents):
            regions = self.region_detector.detect_regions(page_content)
            all_regions.extend(regions)
            page_images.append(self._render_page(pdf_path, page_idx))

        return {
            'regions': all_regions,
            'page_images': page_images,
            'question': row['question'],
            'answer': row['answer'],
            'q_id': row['q_id']
        }
```

#### 5.2 `scripts/train.py`
```python
def train(config):
    # Model
    model = UniDocModel(config)

    # Dataset
    train_dataset = UniDocDataset(config.train_csv, config)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              collate_fn=unidoc_collate_fn)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=config.lr)

    # Training loop
    for epoch in range(config.epochs):
        for batch in train_loader:
            outputs = model(batch)
            loss = compute_loss(outputs, batch['answer'])
            loss.backward()
            optimizer.step()
```

---

## 구현 우선순위 및 일정

| Phase | 모듈 | 우선순위 | 예상 시간 |
|-------|------|----------|-----------|
| 1 | PDF Vector Parsing | 높음 | 3-4일 |
| 2 | Region Grouping | 높음 | 4-5일 |
| 3 | **Unified Region Tokenization** | **최우선** | 5-7일 |
| 4 | Layout-Aware Transformer | 높음 | 3-4일 |
| 5 | LLM Adapter | 중간 | 2-3일 |
| 6 | Training Pipeline | 중간 | 2-3일 |
| 7 | Evaluation & 실험 | 높음 | 3-4일 |

---

## 의존성 추가 (requirements.txt)

```
# 기존
torch>=2.0.0
transformers>=4.30.0
pandas
tqdm
pillow

# 추가
pymupdf>=1.23.0          # PDF vector parsing
pdfplumber>=0.10.0       # 보조 PDF 파싱
opencv-python>=4.8.0     # 이미지 처리
scipy>=1.11.0            # 수치 연산
networkx>=3.0            # 그래프 알고리즘 (region grouping)
einops>=0.7.0            # Tensor 연산
timm>=0.9.0              # Vision models
accelerate>=0.25.0       # 분산 학습
wandb>=0.16.0            # 실험 추적
```

---

## 핵심 구현 포인트 체크리스트

### ⭐ Unified Region Tokenization (가장 중요)

- [ ] Text embedding: region 내 텍스트 → dense vector
- [ ] Visual embedding: region crop → visual feature
- [ ] Layout embedding: normalized 2D coordinates
- [ ] Role embedding: region type encoding
- [ ] **Fusion**: 4가지를 concat → single hidden dim으로 projection
- [ ] **핵심**: 이 과정이 "추출 시점에서 통합"을 구현

### PDF Vector Parsing

- [ ] PyMuPDF로 text runs 추출 (글자별 위치, 폰트 정보)
- [ ] Drawing paths 추출 (선, 사각형 → 표 구조)
- [ ] Embedded images 추출
- [ ] OCR 없이 digital PDF에서 정확한 텍스트 획득

### Region Grouping

- [ ] 문단 grouping (거리, 폰트 일관성 기반)
- [ ] 표 detection (drawing paths의 그리드 구조)
- [ ] Figure-caption association
- [ ] Reading order 결정

### Layout-Aware Attention

- [ ] Spatial bias 계산 (pairwise relative position)
- [ ] Attention score에 bias 추가
- [ ] 인접 region 간 attention 강화

---

## 예상 결과물

1. **UniDoc 모델**: Vector-native unified tokenization 기반 Document VQA 모델
2. **벤치마크 결과**: 4개 데이터셋 (feta_tab, slidevqa, scigraphvqa, spiqa)에서 평가
3. **Ablation Study**: 각 컴포넌트의 기여도 분석
4. **논문**: CVPR/ICCV/ECCV 또는 NeurIPS/ICML 타겟
