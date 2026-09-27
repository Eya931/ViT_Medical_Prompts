# Metrics: AUC, F1, etc.
import torch

def accuracy(preds, labels):
	return (preds.argmax(dim=1) == labels).float().mean().item()
