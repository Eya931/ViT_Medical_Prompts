"""
utils.py
Fonctions utilitaires partagées : affichage d'images, visualisation d'attention maps, sauvegarde de checkpoints.
"""
import torch
import matplotlib.pyplot as plt
import os
from torchvision.utils import save_image

# Affichage d'une image
def display_image(img_tensor):
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    plt.imshow((img * 0.229 + 0.485).clip(0, 1))
    plt.axis('off')
    plt.show()

# Visualisation d'une attention map (exemple simplifié)
def display_attention_map(img_tensor, attn_map):
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    plt.imshow((img * 0.229 + 0.485).clip(0, 1))
    plt.imshow(attn_map, alpha=0.5, cmap='jet')
    plt.axis('off')
    plt.show()

# Génération d'une attention map (token CLS, exemple simplifié)
def generate_attention_map(model, imgs, task):
    # Ici, on suppose que le modèle retourne les attn weights ou on utilise le token CLS
    # À adapter selon l'implémentation réelle
    with torch.no_grad():
        x = model.vit._process_input(imgs)
        x = model.vit.patch_embed(x)
        x = model.prompts[task](x)
        x = model.vit.encoder(x)
        cls_token = x[:, 0, :]
        attn_map = cls_token.abs().mean(dim=1).reshape(14, 14).cpu().numpy()
    return attn_map

# Sauvegarde de checkpoint
def save_checkpoint(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Checkpoint sauvegardé: {path}")
