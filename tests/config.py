"""
Project: Learning Multi-modal Representations by Watching Hundreds of Surgical Video Lectures
-----
Copyright (c) University of Strasbourg, All Rights Reserved.
"""
import torch
import torchvision.transforms as transforms

config = dict(
    dataset_config=[
    dict(
    type='Recognition_frame',
    csv_root='./csvs',
    vid='video%02d.csv'%i,
    video_root='./tmp/tmp',
    transforms=transforms.Compose(
        [
        transforms.Resize((360, 640)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        ),
    ) for i in range(49, 81)
    ],
    model_config = dict(
        type='SurgVLP',
        backbone_img = dict(
            type='img_backbones/ImageEncoder',
            num_classes=768,
            pretrained='imagenet',
            backbone_name='resnet_50',
            img_norm=False
        ),
        backbone_text= dict(
            type='text_backbones/BertEncoder',
            text_bert_type='emilyalsentzer/Bio_ClinicalBERT',
            text_last_n_layers=4,
            text_aggregate_method='sum',
            text_norm=False,
            text_embedding_dim=768,
            text_freeze_bert=False,
            text_agg_tokens=True
        )
    )
)

