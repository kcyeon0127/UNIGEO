"""
Text Embedding: Region의 텍스트를 Qwen2-VL 방식으로 embedding

Qwen2-VL의 텍스트 처리 파이프라인:
1. Tokenizer로 텍스트 → token ids
2. embed_tokens로 token ids → embeddings
3. (선택) 몇 개 layer를 통과시켜 contextualized embedding

출력: hᵀ ∈ ℝ³⁵⁸⁴
"""

import torch
import torch.nn as nn
from typing import Optional, List, Union
from transformers import AutoTokenizer, AutoProcessor


class TextEmbedding(nn.Module):
    """
    Qwen2-VL의 text embedding 방식 사용

    세 가지 모드:
    1. token_only: embed_tokens만 사용 (가장 가벼움)
    2. shallow: embed_tokens + 처음 몇 개 layer
    3. deep: 전체 text encoder 사용 (가장 무거움)
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        mode: str = "token_only",
        num_layers: int = 4,
        pooling: str = "mean",
        max_length: int = 512,
        device: Optional[str] = None,
        load_model: bool = True
    ):
        """
        Args:
            model_name: Qwen2-VL 모델 이름
            mode: "token_only", "shallow", "deep"
            num_layers: shallow 모드에서 사용할 layer 수
            pooling: "mean", "cls", "last"
            max_length: 최대 토큰 길이
            device: 디바이스 (None이면 자동 선택)
            load_model: True면 모델 로드, False면 나중에 set_model로 설정
        """
        super().__init__()

        self.model_name = model_name
        self.mode = mode
        self.num_layers = num_layers
        self.pooling = pooling
        self.max_length = max_length
        self.hidden_size = 3584  # Qwen2-VL 7B hidden size

        # Device 설정
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Tokenizer 로드 (항상 필요)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        # 모델 컴포넌트 (lazy load)
        self.embed_tokens = None
        self.layers = None
        self.norm = None

        if load_model:
            self._load_model_components()

    def _load_model_components(self):
        """Qwen2-VL에서 필요한 컴포넌트만 로드"""
        from transformers import Qwen2VLForConditionalGeneration

        print(f"Loading Qwen2-VL text components ({self.mode} mode)...")

        # 전체 모델 로드
        full_model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

        # Qwen2-VL 버전에 따라 language model 서브모듈 위치가 다를 수 있음
        language_model = getattr(full_model, "model", None)
        if language_model is None:
            language_model = getattr(full_model, "language_model", None)

        if language_model is None:
            raise RuntimeError("Unable to locate language model inside Qwen2-VL checkpoint")

        # embed_tokens 위치 탐색
        self.embed_tokens = getattr(language_model, "embed_tokens", None)
        if self.embed_tokens is None and hasattr(language_model, "model"):
            self.embed_tokens = getattr(language_model.model, "embed_tokens", None)

        if self.embed_tokens is None:
            raise RuntimeError("Qwen2-VL language model does not expose embed_tokens")

        # Transformer layer/norm 모듈 탐색
        transformer = language_model
        if hasattr(language_model, "model") and hasattr(language_model.model, "layers"):
            transformer = language_model.model

        if self.mode in ["shallow", "deep"]:
            layers = getattr(transformer, "layers", None)
            norm = getattr(transformer, "norm", None)

            if layers is None:
                raise RuntimeError("Unable to locate transformer layers for Qwen2-VL language model")

            if self.mode == "shallow":
                self.layers = nn.ModuleList([
                    layers[i]
                    for i in range(min(self.num_layers, len(layers)))
                ])
            else:
                self.layers = layers

            self.norm = norm

        # 메모리 절약을 위해 나머지 삭제
        del full_model
        torch.cuda.empty_cache()

        print(f"Text embedding components loaded: embed_tokens + {len(self.layers) if self.layers else 0} layers")

    def set_model_components(self, embed_tokens: nn.Embedding, layers: Optional[nn.ModuleList] = None, norm: Optional[nn.Module] = None):
        """외부에서 모델 컴포넌트 설정 (메모리 공유용)"""
        self.embed_tokens = embed_tokens
        self.layers = layers
        self.norm = norm

    @torch.no_grad()
    def encode(self, text: str) -> torch.Tensor:
        """
        단일 텍스트를 embedding

        Args:
            text: 입력 텍스트

        Returns:
            hᵀ: (hidden_size,) text embedding
        """
        if self.embed_tokens is None:
            raise RuntimeError("Model components not loaded. Call _load_model_components() or set_model_components() first.")

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=False
        )

        input_ids = inputs.input_ids.to(self.embed_tokens.weight.device)
        attention_mask = inputs.attention_mask.to(self.embed_tokens.weight.device)

        # Embedding
        hidden_states = self.embed_tokens(input_ids)  # (1, seq_len, hidden_size)

        # Layer 통과 (shallow/deep 모드)
        if self.layers is not None:
            for layer in self.layers:
                layer_output = layer(
                    hidden_states,
                    attention_mask=attention_mask.unsqueeze(1).unsqueeze(1),
                )
                hidden_states = layer_output[0]

            if self.norm is not None:
                hidden_states = self.norm(hidden_states)

        # Pooling
        h_T = self._pool(hidden_states, attention_mask)  # (hidden_size,)

        return h_T.float()  # bfloat16 → float32

    @torch.no_grad()
    def encode_batch(self, texts: List[str]) -> torch.Tensor:
        """
        배치 텍스트를 embedding

        Args:
            texts: 입력 텍스트 리스트

        Returns:
            hᵀ: (batch_size, hidden_size) text embeddings
        """
        if self.embed_tokens is None:
            raise RuntimeError("Model components not loaded.")

        # Tokenize with padding
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=True
        )

        input_ids = inputs.input_ids.to(self.embed_tokens.weight.device)
        attention_mask = inputs.attention_mask.to(self.embed_tokens.weight.device)

        # Embedding
        hidden_states = self.embed_tokens(input_ids)

        # Layer 통과
        if self.layers is not None:
            for layer in self.layers:
                layer_output = layer(
                    hidden_states,
                    attention_mask=attention_mask.unsqueeze(1).unsqueeze(1),
                )
                hidden_states = layer_output[0]

            if self.norm is not None:
                hidden_states = self.norm(hidden_states)

        # Pooling
        h_T = self._pool(hidden_states, attention_mask)  # (batch_size, hidden_size)

        return h_T.float()

    def _pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Sequence를 single vector로 pooling

        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len)

        Returns:
            pooled: (batch_size, hidden_size) or (hidden_size,) if batch_size=1
        """
        if self.pooling == "mean":
            # Mean pooling (attention mask 고려)
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
            pooled = sum_embeddings / sum_mask

        elif self.pooling == "cls":
            # 첫 번째 토큰 (CLS-like)
            pooled = hidden_states[:, 0, :]

        elif self.pooling == "last":
            # 마지막 유효 토큰
            seq_lens = attention_mask.sum(dim=1) - 1
            batch_size = hidden_states.shape[0]
            pooled = hidden_states[torch.arange(batch_size), seq_lens]

        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

        # Squeeze if single sample
        if pooled.shape[0] == 1:
            pooled = pooled.squeeze(0)

        return pooled

    def forward(self, text: Union[str, List[str]]) -> torch.Tensor:
        """
        Forward pass (nn.Module 인터페이스)

        Args:
            text: 단일 텍스트 또는 텍스트 리스트

        Returns:
            hᵀ: text embedding(s)
        """
        if isinstance(text, str):
            return self.encode(text)
        else:
            return self.encode_batch(text)


class TextEmbeddingLite(nn.Module):
    """
    경량 Text Embedding (Qwen2-VL 없이 사용 가능)

    사전학습된 sentence transformer나 간단한 embedding 사용
    테스트/디버깅용
    """

    def __init__(
        self,
        hidden_size: int = 3584,
        vocab_size: int = 152064,
        max_length: int = 512
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.max_length = max_length

        # 간단한 embedding + projection
        self.embed = nn.Embedding(vocab_size, 768)
        self.projection = nn.Sequential(
            nn.Linear(768, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )

        # 간단한 tokenizer (character level)
        self.char_to_id = {}

    def encode(self, text: str) -> torch.Tensor:
        """간단한 character-level encoding"""
        # Character to ID
        ids = []
        for char in text[:self.max_length]:
            if char not in self.char_to_id:
                self.char_to_id[char] = len(self.char_to_id) % self.embed.num_embeddings
            ids.append(self.char_to_id[char])

        if not ids:
            ids = [0]

        input_ids = torch.tensor([ids], device=self.embed.weight.device)

        # Embed and pool
        embeddings = self.embed(input_ids)  # (1, seq_len, 768)
        pooled = embeddings.mean(dim=1)  # (1, 768)

        # Project to hidden_size
        h_T = self.projection(pooled).squeeze(0)  # (hidden_size,)

        return h_T

    def forward(self, text: str) -> torch.Tensor:
        return self.encode(text)
