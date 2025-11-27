"""
Visual Embedding: Region 이미지를 Qwen2-VL Vision Encoder로 embedding

Qwen2-VL Vision Encoder 구조:
- ViT with 32 layers
- embed_dim: 1280 (내부)
- output: 3584 (LLM hidden size로 projection)
- patch_size: 14

출력: hᴵ ∈ ℝ³⁵⁸⁴
"""

import torch
import torch.nn as nn
from typing import Optional, List, Union, Tuple
from PIL import Image
import math


class VisualEmbedding(nn.Module):
    """
    Qwen2-VL의 Vision Encoder 사용

    Figure/Table region의 crop된 이미지를 embedding
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        pooling: str = "mean",
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1280 * 28 * 28,
        device: Optional[str] = None,
        load_model: bool = True
    ):
        """
        Args:
            model_name: Qwen2-VL 모델 이름
            pooling: "mean", "cls" - patch embedding pooling 방식
            min_pixels: 최소 픽셀 수
            max_pixels: 최대 픽셀 수
            device: 디바이스
            load_model: True면 모델 로드
        """
        super().__init__()

        self.model_name = model_name
        self.pooling = pooling
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.hidden_size = 3584  # Qwen2-VL 7B output hidden size

        # Device 설정
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Vision encoder (lazy load)
        self.visual = None
        self.processor = None

        if load_model:
            self._load_model_components()

    def _load_model_components(self):
        """Qwen2-VL에서 Vision Encoder만 로드"""
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        print("Loading Qwen2-VL vision encoder...")

        # Processor 로드
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels
        )

        # 전체 모델 로드
        full_model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

        # Vision encoder만 추출
        self.visual = full_model.visual

        # 메모리 절약
        del full_model.model  # LLM 부분 삭제
        torch.cuda.empty_cache()

        print("Vision encoder loaded successfully")

    def set_model_components(self, visual: nn.Module, processor):
        """외부에서 모델 컴포넌트 설정"""
        self.visual = visual
        self.processor = processor

    @torch.no_grad()
    def encode(self, image: Image.Image) -> torch.Tensor:
        """
        단일 이미지를 embedding

        Args:
            image: PIL Image (Region에서 crop한 이미지)

        Returns:
            hᴵ: (hidden_size,) image embedding
        """
        if self.visual is None:
            raise RuntimeError("Vision encoder not loaded.")

        # 이미지가 너무 작으면 resize
        image = self._preprocess_image(image)

        # Process image
        inputs = self.processor(
            images=[image],
            text=[""],
            return_tensors="pt",
            padding=True
        )

        # pixel_values와 image_grid_thw 추출
        pixel_values = inputs.pixel_values.to(
            device=self.visual.device if hasattr(self.visual, 'device') else next(self.visual.parameters()).device,
            dtype=torch.bfloat16
        )
        image_grid_thw = inputs.image_grid_thw.to(pixel_values.device)

        # Vision encoder forward
        # Qwen2VL visual encoder: (batch, num_patches, hidden_size)
        vision_output = self.visual(pixel_values, grid_thw=image_grid_thw)

        # Pooling
        h_I = self._pool(vision_output)  # (hidden_size,)

        return h_I.float()

    @torch.no_grad()
    def encode_batch(self, images: List[Image.Image]) -> torch.Tensor:
        """
        배치 이미지를 embedding

        Args:
            images: PIL Image 리스트

        Returns:
            hᴵ: (batch_size, hidden_size) image embeddings
        """
        if self.visual is None:
            raise RuntimeError("Vision encoder not loaded.")

        # Preprocess
        images = [self._preprocess_image(img) for img in images]

        # Process images
        inputs = self.processor(
            images=images,
            text=["" for _ in images],
            return_tensors="pt",
            padding=True
        )

        pixel_values = inputs.pixel_values.to(
            device=next(self.visual.parameters()).device,
            dtype=torch.bfloat16
        )
        image_grid_thw = inputs.image_grid_thw.to(pixel_values.device)

        # Vision encoder forward
        vision_output = self.visual(pixel_values, grid_thw=image_grid_thw)

        # 배치별로 pooling (이미지마다 patch 수가 다를 수 있음)
        # image_grid_thw: (num_images, 3) - [t, h, w] for each image
        batch_embeddings = []
        current_idx = 0

        for i in range(len(images)):
            t, h, w = image_grid_thw[i]
            num_patches = t * h * w
            image_patches = vision_output[current_idx:current_idx + num_patches]
            pooled = self._pool(image_patches.unsqueeze(0))
            batch_embeddings.append(pooled)
            current_idx += num_patches

        h_I = torch.stack(batch_embeddings, dim=0)  # (batch_size, hidden_size)

        return h_I.float()

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """이미지 전처리"""
        # RGB로 변환
        if image.mode != "RGB":
            image = image.convert("RGB")

        # 최소 크기 보장 (28x28 = 1 patch)
        min_size = 28
        if image.width < min_size or image.height < min_size:
            # 최소 크기로 resize (aspect ratio 유지)
            scale = max(min_size / image.width, min_size / image.height)
            new_width = int(image.width * scale)
            new_height = int(image.height * scale)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        return image

    def _pool(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Patch embeddings를 single vector로 pooling

        Args:
            hidden_states: (1, num_patches, hidden_size) or (num_patches, hidden_size)

        Returns:
            pooled: (hidden_size,)
        """
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.squeeze(0)  # (num_patches, hidden_size)

        if self.pooling == "mean":
            pooled = hidden_states.mean(dim=0)
        elif self.pooling == "cls":
            pooled = hidden_states[0]
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

        return pooled

    def forward(self, image: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
        """Forward pass"""
        if isinstance(image, Image.Image):
            return self.encode(image)
        else:
            return self.encode_batch(image)


class VisualEmbeddingLite(nn.Module):
    """
    경량 Visual Embedding (Qwen2-VL 없이 사용 가능)

    간단한 CNN + projection
    테스트/디버깅용
    """

    def __init__(
        self,
        hidden_size: int = 3584,
        image_size: int = 224
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.image_size = image_size

        # Simple CNN encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        # Projection to hidden_size
        self.projection = nn.Sequential(
            nn.Linear(256, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )

        # Image transform
        self.transform = None

    def _get_transform(self):
        """이미지 변환 함수"""
        if self.transform is None:
            from torchvision import transforms
            self.transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        return self.transform

    def encode(self, image: Image.Image) -> torch.Tensor:
        """단일 이미지 encoding"""
        # RGB 변환
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Transform
        transform = self._get_transform()
        img_tensor = transform(image).unsqueeze(0)  # (1, 3, H, W)
        img_tensor = img_tensor.to(next(self.parameters()).device)

        # Encode
        features = self.encoder(img_tensor)  # (1, 256)
        h_I = self.projection(features).squeeze(0)  # (hidden_size,)

        return h_I

    def forward(self, image: Image.Image) -> torch.Tensor:
        return self.encode(image)


class CLIPVisualEmbedding(nn.Module):
    """
    CLIP Vision Encoder 사용 (대안)

    Qwen2-VL 대신 CLIP ViT 사용
    hidden_size가 다르므로 projection 필요
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        hidden_size: int = 3584,
        pooling: str = "cls",
        device: Optional[str] = None,
        load_model: bool = True
    ):
        super().__init__()

        self.model_name = model_name
        self.hidden_size = hidden_size
        self.pooling = pooling

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # CLIP components
        self.clip_model = None
        self.clip_processor = None
        self.clip_hidden_size = 1024  # CLIP ViT-L hidden size

        # Projection layer (CLIP dim → target dim)
        self.projection = nn.Sequential(
            nn.Linear(self.clip_hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )

        if load_model:
            self._load_model()

    def _load_model(self):
        """CLIP 모델 로드"""
        from transformers import CLIPModel, CLIPProcessor

        print(f"Loading CLIP vision encoder: {self.model_name}")

        self.clip_model = CLIPModel.from_pretrained(self.model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(self.model_name)

        # Vision encoder만 사용
        self.clip_model = self.clip_model.vision_model
        self.clip_model.to(self.device)
        self.clip_model.eval()

        print("CLIP vision encoder loaded")

    @torch.no_grad()
    def encode(self, image: Image.Image) -> torch.Tensor:
        """이미지 encoding"""
        if self.clip_model is None:
            raise RuntimeError("CLIP model not loaded.")

        # RGB 변환
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Process
        inputs = self.clip_processor(images=image, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(self.device)

        # CLIP forward
        outputs = self.clip_model(pixel_values)

        if self.pooling == "cls":
            features = outputs.pooler_output  # (1, clip_hidden_size)
        else:
            features = outputs.last_hidden_state.mean(dim=1)  # (1, clip_hidden_size)

        # Project to target hidden_size
        h_I = self.projection(features).squeeze(0)  # (hidden_size,)

        return h_I

    def forward(self, image: Image.Image) -> torch.Tensor:
        return self.encode(image)
