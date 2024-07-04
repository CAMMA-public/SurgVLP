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
import tqdm

__all__ = ["available_models", "load", "tokenize", "load_dataset"]


_MODELS = {
    "SurgVLP": "https://seafile.unistra.fr/seafhttp/files/0d8e0768-159f-4d05-b547-fa086bf7338d/surgvlp.pth",
} 

_INPUT_RES = {
    "SurgVLP": 224,
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

def _download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        return download_target

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    return download_target

def load(model_config, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", download_root: str = None):
        
    model_name = model_config['type']
    model_path = _download(_MODELS[model_name], download_root or os.path.expanduser("~/.cache/surgvlp"))

    input_size = _INPUT_RES[model_name]

    model = build_algorithm(model_config).to(device)
    model.load_state_dict(torch.load(model_path))
    model = model.eval()

    return model, _transform(input_size)

def load_dataset(config):
    dataset = build_dataset(config)
    return dataset