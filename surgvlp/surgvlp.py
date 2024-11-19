"""
Project: Learning Multi-modal Representations by Watching Hundreds of Surgical Video Lectures
-----
Copyright (c) University of Strasbourg, All Rights Reserved.
"""
import hashlib
import os
import urllib
import warnings
from packaging import version
from typing import Union, List

from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Any, Union
from .codes.models import build_algorithm
from .codes.datasets import build_dataset
import torch

from mmengine.config import Config
import torchvision.transforms as transforms
from tqdm import tqdm
import subprocess
import zipfile

__all__ = ["available_models", "load", "tokenize", "load_dataset"]

_MODELS = {
    "SurgVLP": "https://seafile.unistra.fr/f/93757ace1bfc47248e1e/?dl=1",
    "HecVL": "https://seafile.unistra.fr/f/3b9b9207068a4b03bc2a/?dl=1",
    "PeskaVLP": "https://seafile.unistra.fr/f/65a2b1bf113e428280d0/?dl=1",
} 

_INPUT_RES = {
    "SurgVLP": 224,
    "HecVL": 224,
    "PeskaVLP": 224
} 

def available_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform(n_px):
    return transforms.Compose([
        transforms.Resize((360, 640)),
        transforms.CenterCrop(n_px),
        _convert_image_to_rgb,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def tokenize(
    text: Union[str, List[str]],
    padding: str = 'max_length',
    max_length: int = 77,
    truncation: bool = True,
    model_name: str = 'emilyalsentzer/Bio_ClinicalBERT',
    device: str = 'cpu'
) -> Dict[str, Any]:
    tokenizer_clinical = AutoTokenizer.from_pretrained(model_name)
    ixtoword = {v: k for k, v in tokenizer_clinical.get_vocab().items()}

    if isinstance(text, str):
        text = [text]

    processed_text_tensors = []
    for t in text:
        text_tensors = tokenizer_clinical(
            t,
            return_tensors="pt",
            truncation=truncation,
            padding=padding,
            max_length=max_length,
        )
        text_tensors["sent"] = [ixtoword[ix] for ix in text_tensors["input_ids"][0].tolist()]
        processed_text_tensors.append(text_tensors)

    caption_ids = torch.stack([x["input_ids"] for x in processed_text_tensors])
    attention_mask = torch.stack([x["attention_mask"] for x in processed_text_tensors])
    token_type_ids = torch.stack([x["token_type_ids"] for x in processed_text_tensors])

    # Squeeze if only one text
    if len(text) == 1:
        caption_ids = caption_ids.squeeze(0).to(device)
        attention_mask = attention_mask.squeeze(0).to(device)
        token_type_ids = token_type_ids.squeeze(0).to(device)
    else:
        caption_ids = caption_ids.squeeze().to(device)
        attention_mask = attention_mask.squeeze().to(device)
        token_type_ids = token_type_ids.squeeze().to(device)

    cap_lens = [len([w for w in txt if not w.startswith("[")]) for txt in text]

    return {
        "input_ids": caption_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "cap_lens": cap_lens,
    }


def _download(models: Dict[str, str], key: str, root: str) -> str:
    url = models[key]
    os.makedirs(root, exist_ok=True)
    filename = key + '.zip'
    
    download_target = os.path.join(root, filename)
    
    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")
    
    if os.path.isfile(download_target):
        if zipfile.is_zipfile(download_target):
            with zipfile.ZipFile(download_target, 'r') as zip_ref:
                zip_ref.extractall(root)
        return download_target.replace('.zip', '.pth')
        
    # Using wget to download the file with --content-disposition
    command = ['wget', '--content-disposition', '-P', root, url]
    
    result = subprocess.run(command, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Failed to download file: {result.stderr}")
    
    # Check if the downloaded file is a zip file and unzip it
    if zipfile.is_zipfile(download_target):
        with zipfile.ZipFile(download_target, 'r') as zip_ref:
            zip_ref.extractall(root)
    
    return download_target.replace('.zip', '.pth')

def load(model_config, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", download_root: str = None):
        
    model_name = model_config['type']
    model_path = _download(_MODELS, model_name, download_root or os.path.expanduser("~/.cache/surgvlp"))

    input_size = _INPUT_RES[model_name]

    model = build_algorithm(model_config).to(device)
    model.load_state_dict(torch.load(model_path))
    model = model.eval()

    return model, _transform(input_size)

def load_dataset(config):
    dataset = build_dataset(config)
    return dataset
