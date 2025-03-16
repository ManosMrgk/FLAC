import time
import os
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
from torchvision import transforms
from datasets.mimic_cxr_jpg import MimicCXR
from utils.utils import (
    AverageMeter
)
from models.densenet import DenseNet121

def sanitize_data(data):
    for key, value in data.items():
        if isinstance(value, dict):
            sanitize_data(value)  # To sanitize the nested dictionaries
        elif isinstance(value, torch.Tensor):
            data[key] = value.cpu().numpy() if value.is_cuda else value.numpy()

def initialize_model(models_dir, model_path, class_names):
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"Created directory: {models_dir}", flush=True)

    if os.path.exists(model_path):
        model = torch.load(model_path)
        print("Model loaded from saved state:", model_path, flush=True)
    else:
        print("MODEL NOT FOUND:", model_path)
        model = DenseNet121(num_classes=len(class_names)).cuda()
    return model

def calc_mean_std(dataloader):
    # Initialize mean and std
    sum_mean = torch.zeros(3)
    sum_std = torch.zeros(3)
    num_samples = 0
    with tqdm(dataloader, desc=f'Test calculation') as t:
        for inputs, _, _ in t:
            images = inputs['image']
            batch_samples = images.size(0)  # Batch size
            images = images.view(batch_samples, 3, -1)  # Flatten spatial dimensions
            sum_mean += images.mean(dim=[0, 2])  # Mean per channel
            sum_std += images.std(dim=[0, 2])  # Std per channel
            num_samples += batch_samples
            t.set_postfix()
        mean = sum_mean / num_samples
        std = sum_std / num_samples

    print(f"Mean: {mean.tolist()}")
    print(f"Std: {std.tolist()}")

def main():
    start_time = time.time()
    print("Initialize the dataset", flush=True)
    csv_file='/home/csi22304/FLAC/data/half_meta_data_filtered80.csv'
    print("Initialize the dataset", flush=True)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    print("Using dataset csv:", csv_file)
    class_names = ['No Finding', 'Pleural Effusion', 'Lung Opacity', 'Atelectasis']
    dataset = MimicCXR(
        csv_file=csv_file, root='/home/csi22304/physionet/physionet.org/files/mimic-cxr-jpg/2.0.0/', transform=transform, class_names=class_names, testing=True,
        logo=False, gaussian_noise=False, salt_and_pepper=False, brightness_bands=False, sinusoidal_bands=False, gaussian_smoothing=False, color_inversion=False, noise_intensity=35
    )
    random_state=42
    num_samples = len(dataset)
    indices = list(range(num_samples))
    np.random.seed(random_state)
    np.random.shuffle(indices)
    batch_size = 64

    test_size = len(indices)

    print("Dataset size:", test_size)

    test_dataloader = DataLoader(dataset, batch_size=batch_size)

    print("Calculating mean and standard deviation", flush=True)
    calc_mean_std(test_dataloader)

if __name__ == "__main__":
    main()


