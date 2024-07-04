"""
Project: Learning Multi-modal Representations by Watching Hundreds of Surgical Video Lectures
-----
Copyright (c) University of Strasbourg, All Rights Reserved.
"""
from mmengine import Registry

__all__ = [
    'MODELS',
    'LOSSES',
    'LOOPS',
    'ENGINES'
]

MODELS = Registry('model')
DATASETS = Registry('dataset')
LOSSES = Registry('loss')
ENGINES = Registry('engine')
