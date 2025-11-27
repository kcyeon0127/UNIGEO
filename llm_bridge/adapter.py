"""Bridge absolute embeddings into Qwen2-VL for downstream generation."""

from dataclasses import dataclass
from typing import Optional

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer


@dataclass
class AbsoluteEmbeddingBundle:
    z_text: Optional[torch.Tensor]
    z_image: Optional[torch.Tensor]
    z_layout: torch.Tensor


class QwenVLAdapter:
    def __init__(self, model_name: str = "Qwen/Qwen2-VL-7B-Instruct", z_dim: int = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        language_model = getattr(self.model, "model", None)
        if language_model is None:
            language_model = getattr(self.model, "language_model", None)
        if language_model is None:
            raise RuntimeError("Unable to locate Qwen language model")

        self.embed_tokens = language_model.get_input_embeddings()
        self.hidden_size = self.embed_tokens.embedding_dim
        self.prefix_projector = torch.nn.Linear(z_dim * 3, self.hidden_size)
        self.prefix_projector.to(self.model.device)

    def _concat_modalities(self, bundle: AbsoluteEmbeddingBundle) -> torch.Tensor:
        def _ensure(t: Optional[torch.Tensor], like: torch.Tensor):
            if t is None:
                return torch.zeros_like(like)
            return t

        z_layout = bundle.z_layout
        z_text = _ensure(bundle.z_text, z_layout)
        z_image = _ensure(bundle.z_image, z_layout)
        return torch.cat([z_text, z_image, z_layout], dim=-1)

    def build_inputs(self, bundle: AbsoluteEmbeddingBundle, instruction: str):
        z_cat = self._concat_modalities(bundle).to(self.model.device)
        prefix_embed = self.prefix_projector(z_cat).unsqueeze(0).unsqueeze(1)

        tokens = self.tokenizer(
            instruction,
            return_tensors="pt",
            add_special_tokens=True
        ).to(self.model.device)

        text_embeds = self.embed_tokens(tokens.input_ids)
        inputs_embeds = torch.cat([prefix_embed, text_embeds], dim=1)
        attention_mask = torch.cat([
            torch.ones((tokens.attention_mask.size(0), 1), device=self.model.device, dtype=tokens.attention_mask.dtype),
            tokens.attention_mask
        ], dim=1)

        return inputs_embeds, attention_mask

    def generate(self, bundle: AbsoluteEmbeddingBundle, instruction: str, **gen_kwargs) -> str:
        inputs_embeds, attention_mask = self.build_inputs(bundle, instruction)
        outputs = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=gen_kwargs.get("max_new_tokens", 64),
            do_sample=gen_kwargs.get("do_sample", False)
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
