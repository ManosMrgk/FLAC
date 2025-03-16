import argparse
import datetime
import logging
import os
import time
from pathlib import Path
from torchvision import transforms
import numpy as np
import torch
from PIL import Image
from torch import nn
from flac import flac_loss
from datasets.mimic_cxr_jpg import MimicCXR, get_utk_face
from models.resnet import ResNet18, SimpleCNN
from utils.logging import set_logging
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from utils.utils import (
    AverageMeter,
    MultiDimAverageMeter,
    accuracy,
    load_model,
    pretty_dict,
    save_model,
    set_seed,
)
from tqdm import tqdm

def tensor_to_pil_image(tensor):
    np_array = tensor.cpu().numpy().transpose((1, 2, 0))  # Convert to HWC
    np_array = (np_array * 255).astype(np.uint8)  # Scale to [0, 255]
    return Image.fromarray(np_array)

def denormalize(tensor, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]):
    # Convert mean and std to tensors
    device = tensor.device
    mean = torch.tensor(mean, device=device).view(-1, 1, 1)
    std = torch.tensor(std, device=device).view(-1, 1, 1)

    # Denormalize the image
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)  # Ensure pixel values are within [0, 1]

    return tensor



def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--task", type=str, default="sinusoidal_bands")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bs", type=int, default=64, help="batch_size")
    parser.add_argument("--csv_file", type=str, default='/home/csi22304/mimic_debiasing/data/balanced_test_small.csv') #meta_data_big_with_age.csv')
    parser.add_argument("--root_dir", type=str, default='/home/csi22304/physionet/physionet.org/files/mimic-cxr-jpg/2.0.0/')
    opt = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)

    return opt


def set_model(opt, num_classes=1):
    criterion1 = nn.BCEWithLogitsLoss()
    if opt.task == "race":
        protected_attr_model = "./bias_capturing_classifiers/bcc_race.pth"
    elif opt.task == "age":
        protected_attr_model = "./bias_capturing_classifiers/bcc_age.pth"
    elif opt.task == "gender":
        protected_attr_model = "./bias_capturing_classifiers/bcc_gender.pth"
    elif opt.task == "logo":
        protected_attr_model = "./bias_capturing_classifiers/bcc_logo18.pth"
    elif opt.task == "brightness_bands":
        protected_attr_model = "./bias_capturing_classifiers/bcc_brightness_bands18xnorm.pth"
    elif opt.task == "sinusoidal_bands":
        protected_attr_model = "./bias_capturing_classifiers/bcc_sinusoidal_bands18xnorm.pth"
    elif opt.task == "gaussian_smoothing":
        protected_attr_model = "./bias_capturing_classifiers/bcc_gaussian_smoothing18xnorm.pth"
    elif opt.task == "color_inversion":
        protected_attr_model = "./bias_capturing_classifiers/bcc_color_inversion18xnorm.pth"
    elif opt.task == "salt_and_pepper":
        protected_attr_model = "./bias_capturing_classifiers/bcc_salt_and_pepper18.pth"
    elif opt.task == "gaussian_noise":
        protected_attr_model = "./bias_capturing_classifiers/bcc_gaussian_noise18.pth"
    model = torch.load(protected_attr_model)
    model.cuda()

    return model, criterion1

def test_model(opt, model, test_loader, criterion, device='cuda'):
    best_val_loss = float('inf')
    epochs_no_improve = 0

    with torch.no_grad():

        # Testing phase
        
        running_loss = 0.0
        correct_test = 0
        total_test = 0
        with tqdm(test_loader, desc=f'Test calculation', unit="batch") as t:
            for inputs, _, labels, _ in t:
                inputs, labels = inputs.to(device), labels.to(device)

                
                logits, _ = model(inputs)
                loss = criterion(logits, labels.unsqueeze(1).float())  # Convert labels to float for BCE

                # Update running loss
                running_loss += loss.item() * inputs.size(0)
                preds = torch.sigmoid(logits) > 0.5
                correct_test += (preds == labels.unsqueeze(1)).sum().item()
                total_test += labels.size(0)
                current_loss = running_loss / total_test
                current_accuracy = correct_test / total_test
                t.set_postfix(loss=current_loss, accuracy=current_accuracy)

        epoch_test_loss = running_loss / total_test
        test_acc = correct_test / total_test
        print(f"Test Loss: {epoch_test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

def main():
    start_time = time.time()
    opt = parse_option()
   
    set_seed(opt.seed)

    np.set_printoptions(precision=3)
    torch.set_printoptions(precision=3)

    print("Initialize the dataset", flush=True)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[-0.000774949905462563, 0.0012312569888308644, 0.004699075594544411],
                             std=[0.02031971886754036, 0.020773280411958694, 0.020680958405137062]),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    print("Using dataset csv:", opt.csv_file)
    class_names = ['No Finding', 'Pleural Effusion', 'Lung Opacity', 'Atelectasis']
    target_attribute = opt.task
    test_dataset = MimicCXR(
        csv_file=opt.csv_file, root=opt.root_dir, transform=transform, class_names=class_names, target_attribute=target_attribute
    )

    set_seed(opt.seed)

    test_size = len(test_dataset)

    print("Test size:", test_size, flush=True)

    # Create data loaders
    test_loader = DataLoader(test_dataset, batch_size=opt.bs, num_workers=2, shuffle=True)

    model, criterion = set_model(opt, num_classes=1)

    test_model(opt, model, test_loader, criterion, device='cuda')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info(f"Total testing time: {total_time_str}")



if __name__ == "__main__":
    main()


