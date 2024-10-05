import argparse
import datetime
import logging
import os
import time
from pathlib import Path
from torchvision import transforms
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from datasets.mimic_cxr_jpg import MimicCXR
from models.resnet import ResNet50, ResNet18
from utils.logging import set_logging
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from utils.utils import (
    set_seed,
)

def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="logo-test")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--task", type=str, default="logo")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--bs", type=int, default=64, help="batch_size")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=100)
    parser.add_argument("--csv_file", type=str, default='/home/csi22304/mimic_debiasing/data/balanced_test_small.csv') #logo_test.csv')
    parser.add_argument("--root_dir", type=str, default='/home/csi22304/physionet/physionet.org/files/mimic-cxr-jpg/2.0.0/')
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--test_split", type=float, default=0.2)
    parser.add_argument("--logo", default=False, action="store_true")
    parser.add_argument("--gaussian_noise", default=False, action="store_true")
    parser.add_argument("--salt_and_pepper", default=False, action="store_true")
    parser.add_argument("--noise_intensity", type=int, default=35)
    opt = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)

    return opt

def main():
    start_time = time.time()
    opt = parse_option()

    exp_name = f"flac-mimic_cxr_{opt.task}-{opt.exp_name}18-lr{opt.lr}-alpha{opt.alpha}-bs{opt.bs}-seed{opt.seed}"
    opt.exp_name = exp_name

    output_dir = f"results/{exp_name}"
    save_path = Path(output_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    set_logging(exp_name, "INFO", str(save_path))
    logging.info(f"Set seed: {opt.seed}")
    set_seed(opt.seed)
    logging.info(f"save_path: {save_path}")

    np.set_printoptions(precision=3)
    torch.set_printoptions(precision=3)

    print("Initialize the dataset", flush=True)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    print("Using dataset csv:", opt.csv_file)
    class_names = ['No Finding', 'Pleural Effusion', 'Lung Opacity', 'Atelectasis']
    dataset = MimicCXR(
        csv_file=opt.csv_file, root=opt.root_dir, transform=transform, class_names=class_names, target_attribute=opt.task,
        logo=opt.logo, gaussian_noise=opt.gaussian_noise, salt_and_pepper=opt.salt_and_pepper,noise_intensity=opt.noise_intensity
    )

    num_samples = len(dataset)
    indices = list(range(num_samples))
    set_seed(opt.seed)
    np.random.shuffle(indices)


    test_indices = indices

    test_size = len(test_indices)

    print("Test size:", test_size, flush=True)

     # Create data loaders
    test_loader = DataLoader(dataset, batch_size=opt.bs, sampler=test_indices, num_workers=2)

    protected_net = ResNet18(num_classes=1)
    if opt.task == "race":
        protected_attr_model = "./bias_capturing_classifiers/bcc_race.pth"
    elif opt.task == "age":
        protected_attr_model = "./bias_capturing_classifiers/bcc_age.pth"
    elif opt.task == "gender":
        protected_attr_model = "./bias_capturing_classifiers/bcc_gender.pth"
    elif opt.task == "logo":
        protected_attr_model = "./bias_capturing_classifiers/bcc_logo18.pth"
    protected_net_checkpoint = torch.load(protected_attr_model)
    if isinstance(protected_net_checkpoint, dict) and 'state_dict' in protected_net_checkpoint:
        print("Loading model state_dict.")
        protected_net.load_state_dict(protected_net_checkpoint['state_dict'])
    elif isinstance(protected_net_checkpoint, dict):
        print("Loading model directly from state_dict.")
        protected_net.load_state_dict(protected_net_checkpoint)
    else:
        # If the entire model was saved, load it directly
        print("Loading full model.")
        protected_net = protected_net_checkpoint
    device='cuda'
    protected_net.to(device)
    protected_net.eval()
    num_correct = 0
    total_predictions = 0
    with tqdm(test_loader, desc=f'Testing') as t:
        with torch.no_grad():
            for inputs, _, labels, _ in t:
                inputs, labels = inputs.to(device), labels.to(device)
                logits, _ = protected_net(inputs)
                probs = torch.sigmoid(logits)
                preds = probs > 0.5
                num_correct += (preds == labels.unsqueeze(1)).sum().item()
                total_predictions += labels.size(0)

    print(f"Number of correct classifications: {num_correct}/{total_predictions}")
    

if __name__ == "__main__":
    main()
