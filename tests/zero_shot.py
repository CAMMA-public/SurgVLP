import torch.nn as nn
import argparse
import torch
import clip
from PIL import Image
from mmengine.config import Config
from transformers import AutoTokenizer
from utils import calc_accuracy, calc_f1
import torchmetrics
import numpy as np
import torchvision.transforms as transforms
import surgvlp

def process_text(text):
    tokenizer_clinical = AutoTokenizer.from_pretrained('/gpfswork/rech/okw/ukw13bv/mmsl/biobert_pretrain_output_all_notes_150000')
    ixtoword = {v: k for k, v in tokenizer_clinical.get_vocab().items()}
    if type(text) == str:
        text = [text]

    processed_text_tensors = []
    for t in text:

        text_tensors = tokenizer_clinical(
            t,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=77,
        )
        text_tensors["sent"] = [
            ixtoword[ix] for ix in text_tensors["input_ids"][0].tolist()
        ]
        processed_text_tensors.append(text_tensors)

    caption_ids = torch.stack([x["input_ids"] for x in processed_text_tensors])
    attention_mask = torch.stack(
        [x["attention_mask"] for x in processed_text_tensors]
    )
    token_type_ids = torch.stack(
        [x["token_type_ids"] for x in processed_text_tensors]
    )

    if len(text) == 1:
        caption_ids = caption_ids.squeeze(0).cuda()
        attention_mask = attention_mask.squeeze(0).cuda()#.to(device)
        token_type_ids = token_type_ids.squeeze(0).cuda()
    else:
        caption_ids = caption_ids.squeeze().cuda()
        attention_mask = attention_mask.squeeze().cuda()
        token_type_ids = token_type_ids.squeeze().cuda()

    cap_lens = []
    for txt in text:
        cap_lens.append(len([w for w in txt if not w.startswith("[")]))

    return {
        "input_ids": caption_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "cap_lens": cap_lens,
    }

import matplotlib.pyplot as plt
def save_fig(img_pil, probs, prompts, classes, save_path, title):
    """
    Plots the classification results with the given class labels and probabilities alongside the original image,
    and saves the combined figure to a file.
    
    :param classes: A list of class labels.
    :param probabilities: A list of probabilities corresponding to each class label.
    :param image_path: The file path to the original image.
    :return: The file path where the combined figure is saved.
    """
    # Load the original image
    original_image = img_pil

    # Plot
    fig, axs = plt.subplots(1, 2, figsize=(15, 7))

    # Plot the classification bars on the first subplot
    axs[0].barh(classes, probs, color='skyblue')
    axs[0].set_xlabel('Probability (%)')


    # for i, (prob, prompt) in enumerate(zip(probs, prompts)):
    #     axs[0].text(prob + 1, i, prompt, va='center', ha='right')


    axs[0].set_xlim(0, 100)

    # Display the original image on the second subplot
    axs[1].imshow(original_image)
    axs[1].axis('off')  # Hide the axis


    plt.suptitle(title, fontsize=16)

    plt.tight_layout()
    
    # Save the figure
    plt.savefig(save_path)
    plt.close()  # Close the figure to avoid displaying it in the notebook
    return None


def tensor_to_plt_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Converts a PyTorch tensor representing an image into a format suitable for displaying with matplotlib.
    
    :param tensor: A PyTorch tensor representing an image.
    :return: A numpy array representing the image suitable for matplotlib.
    """
    # Check if tensor is on GPU, and if so, move to CPU
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    # Convert from PyTorch tensor to numpy array
    np_image = tensor.numpy()
    
    # If the tensor represents a batch of images, take the first one
    if np_image.ndim == 4:
        np_image = np_image[0]
    
    # Convert from CHW to HWC format if necessary
    if np_image.shape[0] in (1, 3):  # grayscale or RGB
        np_image = np.transpose(np_image, (1, 2, 0))
    
    # Normalize the image to [0, 1] if it's not already
    # if np_image.min() < 0 or np_image.max() > 1:
    #     np_image = (np_image - np_image.min()) / (np_image.max() - np_image.min())
    
    return np_image

def unnormalize(tensor: torch.Tensor, mean: list, std: list) -> torch.Tensor:
    """
    Un-normalizes a tensor by applying the mean and std.
    
    :param tensor: A PyTorch tensor representing an image that has been normalized.
    :param mean: The mean used for normalization.
    :param std: The standard deviation used for normalization.
    :return: A PyTorch tensor representing the un-normalized image.
    """
    # Replicate mean and std to match the tensor shape
    mean = torch.tensor(mean).reshape(1, -1, 1, 1)
    std = torch.tensor(std).reshape(1, -1, 1, 1)
    
    # Perform un-normalization
    tensor_unnormalized = tensor * std + mean
    
    return tensor_unnormalized


def main(test_loader, model, args):
    class_prompt=args.class_prompt

    model.eval()

    with open(class_prompt) as f:
        lines = f.readlines()
    f.close()

    class_texts_ = [i.replace('\n', '') for i in lines]
    class_texts = process_text(class_texts_)
    text_features = model(None, class_texts, mode='text')['text_emb'].cuda()
    text_features /= text_features.norm(dim=-1, keepdim=True)


    total_acc = []
    total_f1_phase = []
    total_f1_phase_class = []

    with torch.no_grad():
        for test_loader in test_loaders:
            probs_list = []
            label_list = []

            for i, data in enumerate(test_loader): 
                frames = data['video'].cuda() # (1, M, T, C, H, W)
                # B, M, T, C, H, W = frames.shape
                B, C, H, W = frames.shape

                frames = frames.view(-1, C, H, W)
                image_features = model(frames, None, mode='video')['img_emb'] # (B*M*T, D)
                # image_features = torch.mean(image_features, dim=0, keepdim=True) # (1, D)
                image_features /= image_features.norm(dim=-1, keepdim=True)

                probs = (100.0 * image_features @ text_features.T).softmax(dim=-1) # (1, classes)
                labels = data['label'].cuda()

                probs_list.append(probs)
                label_list.append(labels)

                ### save figure

                if args.save_plot:
                    phase_labels = [
                        'Preparation',
                        'Dividing ligament and peritoneum',
                        'Dividing uterine vessels and ligament',
                        'Transecting the vagina',
                        'Specimen removal',
                        'Suturing',
                        'Washing'
                    ]

                    prompts = [
                        '... I insert trocars to patient abdomen cavity',
                        '... I use grasper to hold gallbladder and ...',
                        '... I use clipper to clip the cystic duct and artery then ...',
                        '... I use the hook to dissect the connective tissue ...',
                        '... I put the gallbladder into the specimen bag ',
                        '... suction and irrigation to clear the surgical field ...',
                        '... I grasp the specimen bag and remove ...'
                    ]
                    # assert frames.shape[0] == 1
                    prediction = phase_labels[torch.argmax(probs[0])]
                    gt = phase_labels[labels[0]]

                    frame_unnormalize = unnormalize(frames[0].cpu(), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

                    # import torchvision.utils as utils
                    # utils.save_image(frame_unnormalize, 'tmp.png')
                    # exit()
                    
                    img_pil = tensor_to_plt_image(frame_unnormalize)
                    probs = (probs[0]* 100).tolist() 

                    save_fig(img_pil, probs, prompts, phase_labels, './qualitative/'+str(i)+'_autolapa.png', 'GT: {} | Prediction: {}'.format(gt, prediction))

                    if i == 1500: exit()

            #
            probs_list = torch.cat(probs_list, 0)
            labels = torch.cat(label_list, 0)
            
            acc = calc_accuracy(probs_list, labels)
            print('accuracy: ', acc)
            f1_class, f1_average = calc_f1(probs_list, labels)
            print('f1 average: ', f1_average)
            print('f1 classes: ', f1_class)

            total_acc.append(acc)
            total_f1_phase.append(f1_average)
        print('f1 phase video-wise average ', np.mean(np.asarray(total_f1_phase)))
        print('Acc video-wise average ', np.mean(np.asarray(total_acc)))



def linear_evaluation(
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    num_classes: int
) -> torch.nn.Module:
    # Freeze the pre-trained model's parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Create a linear classifier
    classifier = nn.Linear(1024, num_classes).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(classifier.parameters())

    # Training loop
    model.eval()  # Ensure the model is in evaluation mode
    for epoch in range(1):
        for batch in train_loader:
            inputs = batch['video'].cuda()
            labels = batch['label'].cuda()

            # Forward pass through the pre-trained model to get features
            with torch.no_grad():
                features = model.encode_image(inputs)

            features = features.to(dtype=torch.float32)
            # Forward pass through the classifier
            outputs = classifier(features)
            loss = criterion(outputs, labels)
            print(loss)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Validation can be added here if needed

    return classifier  # Return the trained classifier


def get_args(description='SurgVLP'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--class_prompt', default='./class_prompt.txt', type=str, help='prompt for categories')
    parser.add_argument('--save_plot', default=False, type=bool, help='save plot or not')
    parser.add_argument('--dataset_config', default='./config.py', type=str, help='dataset config')
    parser.add_argument('--batch_size', default=400, type=int, help='batch for testing')
    args = parser.parse_args()
    return args, parser

import torch.distributed as dist
if __name__ == "__main__":

    args, _ = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    configs = Config.fromfile(args.dataset_config)['config']

    if args.save_plot: args.batch_size = 1

    model = surgvlp.load(configs.model_config)


    # state_dict = torch.load('/gpfswork/rech/okw/ukw13bv/mmsl/configs/Hierarchy_SurgVLP_test_5/epoch0160_archive.pth.tar')['state_dict']

    # new_dict = {}
    # for k, v in state_dict.items():
    #     if 'module.' in k:
    #         new_dict[k[7:].replace('visual.model.', 'backbone_img.model.').replace('text_module.model.', 'backbone_text.model.').replace('visual.global_embedder','backbone_img.global_embedder')] = v

    # a, b = model.load_state_dict(new_dict, strict=True)
    # model.eval()

    test_datasets = surgvlp.load_dataset(configs.dataset_config)

    test_loaders = [torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=4
    ) for test_dataset in test_datasets]
    print(args)

    main(test_loaders, model, args)

