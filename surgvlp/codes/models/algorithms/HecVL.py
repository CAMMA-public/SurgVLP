"""
Project: Learning Multi-modal Representations by Watching Hundreds of Surgical Video Lectures
-----
Copyright (c) University of Strasbourg, All Rights Reserved.
"""
import torch as th
import torch.nn.functional as F
import torch.nn as nn
import os
import numpy as np
import re
from ..backbones.img_backbones import *
from ..backbones.text_backbones import *
from ...registry import MODELS

@MODELS.register_module()
class HecVL(nn.Module):
    def __init__(self,
                 backbone_img: dict,
                 backbone_text: dict,
                 neck=None, # Optional[dict] 
                 head= None, # Optional[dict] 
                 pretrained= None, # Optional[str] 
                 ):

        super().__init__()

        self.backbone_img = MODELS.build(backbone_img)
        self.backbone_text = MODELS.build(backbone_text)

        if neck is not None:
            self.neck = MODELS.build(neck)

        if head is not None:
            self.head = MODELS.build(head)

    @property
    def dtype(self):
        return self.backbone_img.model.conv1.weight.dtype

    @property
    def with_neck(self) -> bool:
        """Check if the model has a neck module."""
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_head(self) -> bool:
        """Check if the model has a head module."""
        return hasattr(self, 'head') and self.head is not None

    @property
    def with_target_generator(self) -> bool:
        """Check if the model has a target_generator module."""
        return hasattr(
            self, 'target_generator') and self.target_generator is not None

    def extract_feat_img(self,
                     inputs, # : List[torch.Tensor]
                     ):
        """The forward function to extract features from neck.
        Args:
            inputs (List[torch.Tensor]): The input videos.
        Returns:
            Tuple[torch.Tensor]: visual feature.
        """
        img_emb_g = self.backbone_img(inputs)
        return img_emb_g

    def extract_feat_text(self,ids, attn_mask, token_type):
        """The forward function to extract features from neck.
        Args:
            inputs (List[torch.Tensor]): The input texts.
        Returns:
            Tuple[torch.Tensor]: textual feature.
        """
        text_emb_l, text_emb_g, sents = self.backbone_text(ids, attn_mask, token_type)
        return text_emb_l, text_emb_g, sents

    def loss(self, inputs):
        latent, mask, ids_restore = self.backbone(inputs[0])
        pred = self.neck(latent, ids_restore)
        loss = self.head(pred, inputs[0], mask)
        losses = dict(loss=loss)
        return losses

    def forward(self, inputs_img=None, inputs_text=None, mode= 'all'):
        if inputs_text is not None:
            input_ids = inputs_text['input_ids']
            token_type_ids = inputs_text['token_type_ids']
            attention_masks = inputs_text['attention_mask']

        if mode == 'video':
            feats_img = self.extract_feat_img(inputs_img)
            return {'img_emb': feats_img}
        elif mode == 'text':
            feats_text_local, feats_text_global, sents = self.extract_feat_text(ids=input_ids, attn_mask=attention_masks, token_type=token_type_ids)
            return {'text_emb': feats_text_global}
        elif mode == 'all':
            feats_img = self.extract_feat_img(inputs_img)

            feats_text_local, feats_text_global, sents = self.extract_feat_text(ids=input_ids, attn_mask=attention_masks, token_type=token_type_ids)
            
            return {'img_emb': feats_img, 'text_emb':feats_text_global}

        else:
            raise RuntimeError(f'Invalid mode "{mode}".')
