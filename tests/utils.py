"""
Project: Learning Multi-modal Representations by Watching Hundreds of Surgical Video Lectures
-----
Copyright (c) University of Strasbourg, All Rights Reserved.
"""
import torch
import torchmetrics
from pycm import ConfusionMatrix

def calc_accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    correct = (preds.argmax(dim=1) == labels).sum().item()
    total = labels.numel()
    return correct / total

from sklearn.metrics import f1_score
def calc_f1(preds: torch.Tensor, labels: torch.Tensor) -> float:

    labels_pred = torch.argmax(preds, dim=1).cpu().numpy()
    labels = labels.cpu().numpy()
    cm = ConfusionMatrix(actual_vector=labels, predict_vector=labels_pred)
    print(cm.Overall_ACC, cm.PPV, cm.TPR, cm.classes, cm.F1_Macro)
    f1 = f1_score(labels, labels_pred, average=None)
    f1_average = f1_score(labels, labels_pred, average='macro')
    print(f1_average)
    return f1, f1_average

import numpy as np
import matplotlib.pyplot as plt

# Create the tensor of shape (T, 1) with values between 1 and 7
# T = 1000  # Example length
# tensor = np.random.randint(1, 8, size=(T, 1))

def plot_phases(tensor: np.ndarray, save_name: None):
    # Flatten the tensor to a 1D array
    data = tensor.flatten()

    # Define the colors (one for each unique prediction)
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta']

    # Create a colormap from the unique values
    unique_values = np.unique(data)
    color_map = {val: colors[i % len(colors)] for i, val in enumerate(unique_values)}

    # Assign colors to the data
    bar_colors = [color_map[val.item()] for val in data]

    # Plot the data with different colors
    # plt.figure(figsize=(20, 2))
    plt.bar(range(len(data)), [1]*len(data), color=bar_colors, edgecolor='none', width=1.0)

    # Remove y-axis labels and ticks as they're not relevant for temporal segmentation
    plt.yticks([])

    # Add labels and title
    plt.xlabel('Frame Index')
    plt.title('Temporal Segmentation Plot')

    if save_name:
        plt.savefig(save_name, bbox_inches='tight')

    plt.show()



import torch.nn.functional as F
# def is_semantically_close(embeddings: torch.Tensor, window_size: int, threshold: float) -> torch.Tensor:
#     """
#     Determine if each embedding vector is semantically close to the prior embedding vectors
#     using moving average and cosine similarity.

#     :param embeddings: Sequence of embedding vectors (2D tensor of shape [num_embeddings, embedding_dim]).
#     :param window_size: Number of prior embeddings to consider for the moving average.
#     :param threshold: Cosine similarity threshold to determine closeness.
#     :return: Boolean tensor indicating if each embedding is semantically close to prior embeddings.
#     """
#     num_embeddings, embedding_dim = embeddings.shape

#     # Compute the moving averages using convolution
#     kernel = torch.ones((768, 768, window_size)) / window_size
#     kernel = kernel.cuda()
#     moving_avg = F.conv1d(embeddings.unsqueeze(0).transpose(1, 2), kernel, padding=window_size-1)

#     moving_avg = moving_avg.squeeze(0).transpose(0, 1)[:num_embeddings] # (T, D)

#     # Normalize embeddings and moving averages
#     embeddings_norm = embeddings # (T, D)
#     moving_avg_norm = moving_avg

#     # Compute cosine similarity
#     cos_sim = F.cosine_similarity(embeddings_norm, moving_avg_norm) # (T, 1)

#     # Determine if similarity exceeds threshold
#     close_flags = cos_sim > threshold

#     return cos_sim, close_flags



def is_semantically_close(embeddings: torch.Tensor, window_size: int, threshold: float) -> torch.Tensor:
    """
    Determine if each embedding vector is semantically close to the prior embedding vectors
    using moving average and cosine similarity.

    :param embeddings: Sequence of embedding vectors (2D tensor of shape [num_embeddings, embedding_dim]).
    :param window_size: Number of prior embeddings to consider for the moving average.
    :param threshold: Cosine similarity threshold to determine closeness.
    :return: Boolean tensor indicating if each embedding is semantically close to prior embeddings.
    """
    num_embeddings, embedding_dim = embeddings.shape

    # Initialize tensor to store cosine similarity and close flags
    cos_sim = torch.zeros(num_embeddings)
    close_flags = torch.zeros(num_embeddings, dtype=torch.bool)

    # Compute moving averages and cosine similarities
    moving_agg_embeddings = []
    moving_agg_embeddings_frame = []
    for i in range(num_embeddings):
        # Compute moving average using average pooling
        start_idx = max(0, i - window_size + 1)
        end_idx = min(i + 1, num_embeddings)
        window_embeddings = embeddings[start_idx:end_idx]
        moving_avg = torch.mean(window_embeddings, dim=0, keepdim=True)

        moving_agg_embeddings.append(moving_avg)
        moving_agg_embeddings_frame.append(window_embeddings.unsqueeze(0))

        # Compute cosine similarity
        cos_sim[i] = F.cosine_similarity(embeddings[i].unsqueeze(0), moving_avg)

        # Determine if similarity exceeds threshold
        close_flags[i] = cos_sim[i] > threshold

    moving_agg_embeddings = torch.cat(moving_agg_embeddings, 0)
    moving_agg_embeddings_frame = torch.cat(moving_agg_embeddings_frame, 0) # (T, window_size, D)

    return cos_sim, close_flags, moving_agg_embeddings, moving_agg_embeddings_frame


import numpy as np
import matplotlib.pyplot as plt

def plot_lines(data: np.ndarray, save_path: str = None, colors=None) -> None:
    len_t, classes = data.shape

    # Generate a time axis
    time = np.arange(len_t)

    # Plot each class with a different color
    for i in range(classes):
        plt.plot(time, data[:, i], color=colors[i]) # , label=f'Class {i+1}'
    
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Line Plot of Classes over Time')
    plt.legend()

    if save_path:
        plt.savefig(save_path)

    plt.show()


def plot_cos_sim_and_flags(cos_sim: torch.Tensor, close_flags: torch.Tensor, threshold: float, save_path: str = None):
    """
    Plot the cosine similarity values and close flags, and save the plots to a file if save_path is provided.

    :param cos_sim: Tensor of cosine similarity values.
    :param close_flags: Tensor of boolean flags indicating closeness.
    :param threshold: Cosine similarity threshold used to determine closeness.
    :param save_path: File path to save the plot. If None, the plot will not be saved.
    """
    plt.figure(figsize=(12, 6))

    # Plot cosine similarity
    plt.subplot(2, 1, 1)
    plt.plot(cos_sim.numpy(), label='Cosine Similarity')
    plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
    plt.title('Cosine Similarity')
    plt.legend()

    # Plot close flags
    plt.subplot(2, 1, 2)
    plt.plot(close_flags.numpy(), label='Close Flags', color='g')
    plt.title('Close Flags')
    plt.legend()

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def viz(cos_sim: torch.Tensor, 
        close_flags: torch.Tensor, 
        phase_pred_prob: torch.Tensor, 
        phase_pred_prob_moving: torch.Tensor, 
        phase_gt: torch.Tensor, 
        threshold: float, 
        save_path: str = None):

    """
    Plot the cosine similarity values and close flags, and save the plots to a file if save_path is provided.

    :param cos_sim: Tensor of cosine similarity values.
    :param close_flags: Tensor of boolean flags indicating closeness.
    :param threshold: Cosine similarity threshold used to determine closeness.
    :param save_path: File path to save the plot. If None, the plot will not be saved.
    """
    # Define the colors (one for each unique prediction)
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta']

    plt.figure(figsize=(18, 18))

    # Plot cosine similarity
    plt.subplot(7, 1, 1)
    plt.plot(cos_sim.numpy(), label='Cosine Similarity')
    plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
    plt.title('Cosine Similarity')
    plt.legend()

    # Plot close flags
    plt.subplot(7, 1, 2)
    plt.plot(close_flags.numpy(), label='Close Flags', color='g')
    plt.title('Close Flags')
    plt.legend()


    # Plot phase_gts
    plt.subplot(7, 1, 3)
    plot_phases(phase_gt, save_name=None)
    plt.title('GT Phase')
    plt.legend()

    pred_list = torch.argmax(phase_pred_prob, dim=-1, keepdim=False)
    # Plot predicted phases
    plt.subplot(7, 1, 4)
    plot_phases(pred_list, save_name=None)
    plt.title('Predicted Phase')
    plt.legend()

    # Plot phase_pred using plot_lines
    plt.subplot(7, 1, 5)
    plot_lines(phase_pred_prob.numpy(), save_path=None, colors=colors)  # Save path is handled by the main function
    plt.title('Predicted Phase Changing')


    pred_list_moving = torch.argmax(phase_pred_prob_moving, dim=-1, keepdim=False)
    # Plot predicted phases with moving average
    plt.subplot(7, 1, 6)
    plot_phases(pred_list_moving, save_name=None)
    plt.title('Predicted Phase moving average')
    plt.legend()

    # Plot phase_pred using plot_lines with moving average
    plt.subplot(7, 1, 7)
    plot_lines(phase_pred_prob_moving.numpy(), save_path=None, colors=colors)  # Save path is handled by the main function
    plt.title('Predicted Phase Changing moving average')


    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()