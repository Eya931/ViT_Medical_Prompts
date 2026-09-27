# models/vit_backbone.py
import torch
import torch.nn as nn
from .visual_prompt import VisualPromptViT

class MultiTaskViT(nn.Module):
    def __init__(self, num_prompts=10):
        super().__init__()

        # ViT avec prompts multi-tâches
        self.vit_prompt = VisualPromptViT(
            num_tasks=1,
            num_prompts=num_prompts,
            embed_dim=768,
            num_classes_list=[7]
        )

    def forward(self, x, task_name="ham10000"):
        # On supporte pour l'instant que ham10000
        logits, embeddings = self.vit_prompt(x, task_id=0)
        return logits, embeddings
