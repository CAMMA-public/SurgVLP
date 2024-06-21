# SurgVLP_install

## Usage

First, create a anaconda environment and [install PyTorch](https://pytorch.org/get-started/locally/) (or later) and torchvision, as well as small additional dependencies, and then install this repo as a Python package. On a CUDA GPU machine, the following will do the trick:

```bash
$ pip install -r requirements.txt
$ pip install ftfy regex tqdm
$ pip install git+https://github.com/Flaick/SurgVLP_install.git
```

```python
import torch
import surgvlp
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = surgvlp.load('configs/model_config.py', device=device)

image = preprocess(Image.open("SurgVLP.png")).unsqueeze(0).to(device)
text = surgvlp.tokenize(['This is preparation phase', 'This is clipcutting phase'], device=device)

with torch.no_grad():
    output_dict = model(image, text , mode='all')

    image_embeddings = output_dict['img_emb']
    text_embeddings= output_dict['text_emb']

    image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)
    text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)

    logits_per_image = (100.0 * image_embeddings @ text_embeddings.T).softmax(dim=-1)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)
```


## API

The SurgVLP module `surgvlp` provides the following methods:

#### `surgvlp.available_models()`

Returns the names of the available pre-trained surgical vision-language models.

#### `surgvlp.load(name, device=...)`

Returns the model and the TorchVision transform needed by the model, specified by the model name returned by `surgvlp.available_models()`. It will download the model as necessary. The `name` argument can also be a path to a local checkpoint.

The device to run the model can be optionally specified, and the default is to use the first CUDA device if there is any, otherwise the CPU.

#### `surgvlp.tokenize(text: Union[str, List[str]], max_length=77)`

Returns a LongTensor containing tokenized sequences of given text input(s). This can be used as the input to the model

---

The model returned by `surgvlp.load()` supports the following methods:

#### `model.encode_image(image: Tensor, text: None, mode='video')`

Given a batch of images, returns the image features encoded by the vision portion of the SurgVLP model.

#### `model.encode_text(image: None, text: Tensor, mode='text)`

Given a batch of text tokens, returns the text features encoded by the language portion of the SurgVLP model.

#### `model(image: Tensor, text: Tensor, mode='all')`

Given a batch of images and a batch of text tokens, returns the image and textual embeddings.

