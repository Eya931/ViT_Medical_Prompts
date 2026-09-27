# models/visual_prompt.py
import torch
import torch.nn as nn
import timm

class VisualPromptViT(nn.Module):
    def __init__(self, num_tasks=1, num_prompts=10, embed_dim=768, num_classes_list=[7]):
        super().__init__()

        # ViT pré-entraîné
        self.vit = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,
            num_classes=0
        )

        self.embed_dim = embed_dim

        # Prompts learnables pour chaque tâche
        self.prompts = nn.ParameterList([
            nn.Parameter(torch.randn(1, num_prompts, embed_dim))
            for _ in range(num_tasks)
        ])

        # Têtes de classification
        self.heads = nn.ModuleList([
            nn.Linear(embed_dim, nc) for nc in num_classes_list
        ])

    def forward(self, x, task_id=0):
        B = x.size(0)
        x = self.vit.patch_embed(x)  # [B, N_patches, D]

        cls_token = self.vit.cls_token.expand(B, -1, -1)  # [B,1,D]
        prompt = self.prompts[task_id].expand(B, -1, -1)  # [B,num_prompts,D]

        # Concat CLS + prompt + patches
        x = torch.cat([cls_token, prompt, x], dim=1)  # [B, 1+num_prompts+N_patches, D]

        # Position embedding : on ajoute 0 aux prompts
        # pos_embed: [1,197,D]
        pos_embed_cls = self.vit.pos_embed[:, :1, :]  # CLS
        pos_embed_patches = self.vit.pos_embed[:, 1:, :]  # patches

        # Prompts n'ont pas de pos_embed pré-entraîné, on met 0
        pos_embed_prompt = torch.zeros(1, self.prompts[task_id].size(1), self.embed_dim, device=x.device)

        # Concat CLS + prompts + patches
        pos_embed_full = torch.cat([pos_embed_cls, pos_embed_prompt, pos_embed_patches], dim=1)
        x = x + pos_embed_full

        x = self.vit.pos_drop(x)
        for blk in self.vit.blocks:
            x = blk(x)
        x = self.vit.norm(x)

        cls_output = x[:, 0]
        logits = self.heads[task_id](cls_output)
        return logits, x
