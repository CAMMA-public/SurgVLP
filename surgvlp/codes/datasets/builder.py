# Copyright (c) OpenMMLab. All rights reserved.
from ..registry import DATASETS


def build_dataset(cfg):
    """Build dataset."""
    return DATASETS.build(cfg)