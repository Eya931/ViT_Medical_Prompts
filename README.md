# ViT-Medical-Prompts

Projet de diagnostic médical multi-tâches basé sur Vision Transformer (ViT) et Visual Prompt Learning.


## Nouvelle architecture modulaire

### data/
- `download_datasets.py` : Scripts de téléchargement des datasets
- `chestxray_loader.py` : DataLoader ChestX-ray14
- `brats_loader.py` : DataLoader BraTS2020
- `ham10000_loader.py` : DataLoader HAM10000

### models/
- `vit_backbone.py` : ViT pré-entraîné
- `visual_prompt.py` : Implémentation des prompts visuels
- `multi_task_head.py` : Têtes de classification/segmentation
- `adapter_layers.py` : Adapters (baseline)

### training/
- `train_baseline.py` : Fine-tuning complet
- `train_prompt.py` : Prompt learning
- `train_adapter.py` : Adapters

### evaluation/
- `metrics.py` : AUC, F1, etc.
- `interpretability.py` : Attention maps, Grad-CAM

### experiments/
- `compare_methods.py` : Comparaison des méthodes

### notebooks/
- `01_data_exploration.ipynb`, `02_baseline_results.ipynb`, `03_prompt_analysis.ipynb`


## Exécution indépendante
Chaque module peut être testé séparément :
- Lancer les scripts du dossier `data/` pour tester le téléchargement et les DataLoaders.
- Lancer les modules du dossier `models/` pour vérifier le forward pass et les prompts.
- Lancer les scripts du dossier `training/` pour entraîner sur un mini-batch.
- Lancer les scripts du dossier `evaluation/` pour prédire et visualiser une attention map.


## Visualisation
Les fonctions de visualisation sont dans `evaluation/interpretability.py`.

## Dépendances
- Python >= 3.8
- torch
- torchvision
- matplotlib
- requests

Installez les dépendances avec :
```
pip install torch torchvision matplotlib requests
```

## Remarques
- Les datasets complets doivent être téléchargés via Kaggle ou les liens officiels.
- Les modules sont commentés pour faciliter la prise en main par des débutants.
