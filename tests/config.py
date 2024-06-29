import torch
import torchvision.transforms as transforms

config = dict(
    dataset_config=[
    dict(
    type='Recognition_frame',
    csv_root='/gpfswork/rech/okw/ukw13bv/mmsl/csv/cholec80/csvs',
    vid='video%02d.csv'%i,
    video_root='/gpfsscratch/rech/okw/ukw13bv/cholec80/frames_output',
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
        type='MVNet',
        backbone_img = dict(
            type='img_backbones/ImageEncoder',
            # type='img_backbones/ImageEncoder_CLIPVISUAL',
            num_classes=768,
            pretrained='imagenet', # imagenet/ssl/random
            backbone_name='resnet_50', 
            # backbone_name='resnet_50_clip' 
            img_norm=False,
        ),
        backbone_text= dict(
            type='text_backbones/BertEncoder',
            text_bert_type='/gpfswork/rech/okw/ukw13bv/mmsl/biobert_pretrain_output_all_notes_150000',
            text_last_n_layers=4,
            text_aggregate_method='sum',
            text_norm=False,
            text_embedding_dim=768,
            text_freeze_bert=False,
            text_agg_tokens=True
        )
    )
)

