# Evaluation script for ChestX-ray14
import torch
from models.vit_backbone import MultiTaskViT
from data.chestxray_loader import ChestXray14Dataset
from torch.utils.data import DataLoader
from evaluation.metrics import accuracy
from evaluation.interpretability import generate_attention_map, display_attention_map

if __name__ == "__main__":
    dataset = ChestXray14Dataset("./chestxray14_test")
    loader = DataLoader(dataset, batch_size=2)
    model = MultiTaskViT()
    model.eval()
    with torch.no_grad():
        for imgs, labels in loader:
            outputs = model(imgs, "chestxray")
            acc = accuracy(outputs, labels)
            print(f"Accuracy: {acc:.2f}")
            attn_map = generate_attention_map(model, imgs, "chestxray")
            display_attention_map(imgs[0], attn_map)
            break
