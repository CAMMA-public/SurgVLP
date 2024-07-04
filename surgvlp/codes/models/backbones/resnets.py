"""
Project: Learning Multi-modal Representations by Watching Hundreds of Surgical Video Lectures
-----
Copyright (c) University of Strasbourg, All Rights Reserved.
"""
import torch as th
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torchvision import models as models_2d

class Identity(nn.Module):
    """Identity layer to replace last fully connected layer"""

    def forward(self, x):
        return x



################################################################################
# ResNet Family
################################################################################
import torch

def resnet_18(pretrained=True):
    model = models_2d.resnet18(pretrained=pretrained)
    feature_dims = model.fc.in_features
    model.fc = Identity()
    return model, feature_dims, 1024


def resnet_34(pretrained=True):
    model = models_2d.resnet34(pretrained=pretrained)
    feature_dims = model.fc.in_features
    model.fc = Identity()
    return model, feature_dims, 1024


def resnet_50(pretrained='imagenet'):
    if pretrained=='imagenet':
        model = models_2d.resnet50(pretrained=True)
    elif pretrained=='random':
        model = models_2d.resnet50(pretrained=False)
    elif pretrained=='ssl':
        model = models_2d.resnet50(pretrained=False)
        model.load_state_dict(torch.load('/gpfsdswork/projects/rech/okw/ukw13bv/rendezvous-main/pytorch/converted_vissl_moco_r50_checkpoint.torch'), strict=True)
    feature_dims = model.fc.in_features
    model.fc = Identity()
    return model, feature_dims, 1024

def resnet_101(pretrained=True):
    model = models_2d.resnet101(pretrained=pretrained)
    feature_dims = model.fc.in_features
    model.fc = Identity()
    return model, feature_dims, 1024


resnet_dict = {
    'resnet_18': resnet_18,
    'resnet_34': resnet_34,
    'resnet_50': resnet_50,
    'resnet_101': resnet_101,
}

################################################################################
# DenseNet Family
################################################################################


def densenet_121(pretrained=True):
    model = models_2d.densenet121(pretrained=pretrained)
    feature_dims = model.classifier.in_features
    model.classifier = Identity()
    return model, feature_dims, None


def densenet_161(pretrained=True):
    model = models_2d.densenet161(pretrained=pretrained)
    feature_dims = model.classifier.in_features
    model.classifier = Identity()
    return model, feature_dims, None


def densenet_169(pretrained=True):
    model = models_2d.densenet169(pretrained=pretrained)
    feature_dims = model.classifier.in_features
    model.classifier = Identity()
    return model, feature_dims, None


################################################################################
# ResNextNet Family
################################################################################


def resnext_50(pretrained=True):
    model = models_2d.resnext50_32x4d(pretrained=pretrained)
    feature_dims = model.fc.in_features
    model.fc = Identity()
    return model, feature_dims, None


def resnext_100(pretrained=True):
    model = models_2d.resnext101_32x8d(pretrained=pretrained)
    feature_dims = model.fc.in_features
    model.fc = Identity()
    return model, feature_dims, None
