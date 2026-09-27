# training/train_baseline.py
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data.ham10000_loader import HAM10000Dataset
from models.vit_backbone import MultiTaskViT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device utilisé: {device}")

os.makedirs("saved_models", exist_ok=True)

# Dataset
dataset = HAM10000Dataset(
    "data/HAM10000/HAM10000_metadata.csv",
    "data/HAM10000/images"
)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Modèle
model = MultiTaskViT(num_prompts=10).to(device)

# Optimizer & Loss
params = list(model.vit_prompt.prompts.parameters()) + list(model.vit_prompt.heads.parameters())
optimizer = optim.Adam(params, lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

# Training
model.train()
for epoch in range(3):  # test rapide
    epoch_loss = 0
    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs, _ = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print(f"Epoch {epoch} Batch {batch_idx} Loss: {loss.item():.4f}")
    print(f"Epoch {epoch} Loss moyenne: {epoch_loss/len(loader):.4f}")

# Sauvegarder
torch.save(model.state_dict(), "saved_models/baseline_ham10000.pth")
print("Modèle sauvegardé ✅")
