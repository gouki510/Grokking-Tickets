import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os


def visualize_weight_distribution(model, save_path, epoch, is_div=False):
    """
    Visualizes weight distribution of model.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for name, param in model.named_parameters():
        if 'weight' in name:
            plt.figure()
            plt.hist(param.detach().cpu().numpy().flatten(), bins=100)
            plt.title(f"{name} at epoch {epoch}")
            plt.savefig(save_path + f"{name}_{epoch}.png")
            plt.close()

def visualize_weight(model, save_path, epoch, is_div=False):
    """
    Visualizes weight distribution of model.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for name, param in model.named_parameters():
        if 'weight' in name:
            plt.figure()
            plt.imshow(param.detach().cpu().numpy())
            plt.title(f"{name} at epoch {epoch}")
            plt.savefig(save_path + f"{name}_{epoch}.png")
            plt.close()

def lines(model, save_path, epoch, is_div=False):
    """
    Visualizes weight distribution of model.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for name, param in model.named_parameters():
        if 'weight' in name:
            plt.figure()
            plt.plot(param.detach().cpu().numpy().flatten())
            plt.title(f"{name} at epoch {epoch}")
            plt.savefig(save_path + f"{name}_{epoch}.png")
            plt.close()

