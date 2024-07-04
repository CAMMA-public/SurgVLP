"""
Project: Learning Multi-modal Representations by Watching Hundreds of Surgical Video Lectures
-----
Copyright (c) University of Strasbourg, All Rights Reserved.
"""
from .algorithms import *
from .backbones import *

from .builder import (ALGORITHMS, BACKBONES, HEADS, LOSSES, MEMORIES, NECKS,
                      build_algorithm, build_backbone, build_head, build_loss,
                      build_memory, build_neck)