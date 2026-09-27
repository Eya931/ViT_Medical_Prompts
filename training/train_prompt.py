# training/train_prompt.py
from torch.utils.data import DataLoader, random_split
from data.ham10000_loader import HAM10000Dataset
from torchvision import transforms

task_name = "ham10000"

dataset = HAM10000Dataset(
    "data/HAM10000/HAM10000_metadata.csv",
    "data/HAM10000/images",
    transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
)

num_classes = 7

# Split train / val
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
