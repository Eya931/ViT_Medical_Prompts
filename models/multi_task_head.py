# Multi-task heads for classification/segmentation
# gère les têtes

import torch.nn as nn

def get_heads(embed_dim):
	return nn.ModuleDict({
		"chestxray": nn.Linear(embed_dim, 14),
		"brats": nn.Conv2d(embed_dim, 2, kernel_size=1),
		"ham10000": nn.Linear(embed_dim, 7)
	})
