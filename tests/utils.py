"""
Project: Learning Multi-modal Representations by Watching Hundreds of Surgical Video Lectures
-----
Copyright (c) University of Strasbourg, All Rights Reserved.
"""
import torch
import torchmetrics
from pycm import ConfusionMatrix
import numpy as np
import torch.nn.functional as F

def calc_accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    correct = (preds.argmax(dim=1) == labels).sum().item()
    total = labels.numel()
    return correct / total

from sklearn.metrics import f1_score
def calc_f1(preds: torch.Tensor, labels: torch.Tensor) -> float:

    labels_pred = torch.argmax(preds, dim=1).cpu().numpy()
    labels = labels.cpu().numpy()
    cm = ConfusionMatrix(actual_vector=labels, predict_vector=labels_pred)
    print(cm.Overall_ACC, cm.PPV, cm.TPR, cm.classes, cm.F1_Macro)
    f1 = f1_score(labels, labels_pred, average=None)
    f1_average = f1_score(labels, labels_pred, average='macro')
    print(f1_average)
    return f1, f1_average