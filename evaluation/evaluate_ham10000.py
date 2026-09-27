# evaluation/evaluate_ham10000.py
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
from data.ham10000_loader import HAM10000Dataset
from models.vit_backbone import MultiTaskViT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device utilisé: {device}")

# Dataset
dataset = HAM10000Dataset(
    "data/HAM10000/HAM10000_metadata.csv",
    "data/HAM10000/images"
)
loader = DataLoader(dataset, batch_size=8, shuffle=False)

# Modèle
model = MultiTaskViT(num_prompts=10)
model.load_state_dict(torch.load("saved_models/baseline_ham10000.pth", map_location=device))
model = model.to(device)
model.eval()

all_preds, all_labels = [], []

with torch.no_grad():
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs, _ = model(imgs)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

acc = accuracy_score(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds)

print(f"Accuracy: {acc:.4f}")
print("Matrice de confusion:")
print(cm)
