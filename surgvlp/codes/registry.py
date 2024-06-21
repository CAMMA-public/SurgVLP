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
