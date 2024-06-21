config = dict(
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