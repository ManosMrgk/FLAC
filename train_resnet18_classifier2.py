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
    parser.add_argument("--exp_name", type=str, default="classifier18")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--task", type=str, default="salt_and_pepper")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--bs", type=int, default=64, help="batch_size")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=1000)
    parser.add_argument("--csv_file", type=str, default='data/half_meta_data_filtered.csv') #meta_data_big_with_age.csv')
    parser.add_argument("--root_dir", type=str, default='/home/csi22304/physionet/physionet.org/files/mimic-cxr-jpg/2.0.0/')
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--test_split", type=float, default=0.2)
    opt = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)

    return opt


def set_model(opt, num_classes=1):
    criterion1 = nn.BCEWithLogitsLoss()
    model = ResNet18(num_classes=num_classes, pretrained=True).cuda()
    #model = SimpleCNN(num_classes=1).cuda()
    print(model)
    if True:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.extractor[6:].parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True

    return model, criterion1

def train_model(opt, model, save_path, train_loader, val_loader, criterion, optimizer, scheduler, patience=5, device='cuda'):
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(opt.epochs):

        print(f"Epoch {epoch+1}/{opt.epochs}")

        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{opt.epochs}', unit="batch") as t:
            for inputs, _, labels, _ in t:
                inputs, labels = inputs.to(device), labels.to(device)

                if False:
                    for i in range(len(inputs)):
                        pil_image =  tensor_to_pil_image(denormalize(inputs[i]))
                        pil_image.save(f'{save_path}/checkpoints/sample_image{labels[i]}_{i}.png')
                    exit()

                # Forward pass
                optimizer.zero_grad()
                logits, _ = model(inputs)
                loss = criterion(logits, labels.unsqueeze(1).float())  # Convert labels to float for BCE

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                # Update running loss
                running_loss += loss.item() * inputs.size(0)
                preds = torch.sigmoid(logits) > 0.5
                correct_train += (preds == labels.unsqueeze(1)).sum().item()
                total_train += labels.size(0)
                current_loss = running_loss / total_train
                current_accuracy = correct_train / total_train
                t.set_postfix(loss=current_loss, accuracy=current_accuracy)

        epoch_train_loss = running_loss / total_train
        train_acc = correct_train / total_train

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        #scheduler.step()

        with torch.no_grad():
            for inputs, _, labels, _ in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                logits, _ = model(inputs)
                loss = criterion(logits, labels.unsqueeze(1).float())

                running_val_loss += loss.item() * inputs.size(0)
                probs = torch.sigmoid(logits)
                preds = probs > 0.5
                correct_val += (preds == labels.unsqueeze(1)).sum().item()
                # if total_val < 10:
                #     print("Logits:", logits[:10])
                #     print("Probabilities:", probs[:10])
                #     print("Predictions:", preds[:10])
                #     print("Labels:", labels[:10])
                total_val += labels.size(0)
        epoch_val_loss = running_val_loss / total_val
        val_acc = correct_val / total_val
        if scheduler is not None:
            scheduler.step(epoch_val_loss)
        print(f"Train Loss: {epoch_train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        print(f"Val Loss: {epoch_val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

        # Save the best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model, 'best_model.pth')
            print("Best model saved!")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping after {epoch+1} epochs")
            break

    save_file = save_path / "checkpoints" / f"last_classifier"
    # save_model(model, optimizer, opt, opt.epochs, save_file)
    save_checkpoint(model, optimizer, scheduler, save_file)
    protected_attr_model = None
    if opt.task == "race":
        protected_attr_model = "./bias_capturing_classifiers/bcc_race.pth"
    elif opt.task == "age":
        protected_attr_model = "./bias_capturing_classifiers/bcc_age.pth"
    elif opt.task == "gender":
        protected_attr_model = "./bias_capturing_classifiers/bcc_gender.pth"
    elif opt.task == "logo":
        protected_attr_model = "./bias_capturing_classifiers/bcc_logo18.pth"
    elif opt.task == "brightness_bands":
        protected_attr_model = "./bias_capturing_classifiers/bcc_brightness_bands18.pth"
    else:
        protected_attr_model = f"./bias_capturing_classifiers/bcc_{opt.task}18.pth"

    torch.save(model, protected_attr_model)

def save_checkpoint(model, optimizer, lr_scheduler, path):
    path = str(path)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict()
    }, path+"_state.pth")
    print("Saving model at:", path+".pth")
    torch.save(model, path+".pth")

def load_checkpoint(path, model, optimizer, lr_scheduler):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

    return model, optimizer, lr_scheduler

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = time.time()
    opt = parse_option()

    exp_name = f"flac-mimic_cxr_{opt.task}-{opt.exp_name}-lr{opt.lr}-alpha{opt.alpha}-bs{opt.bs}-seed{opt.seed}"
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
    target_attribute = opt.task
    train_dataset = MimicCXR(
        csv_file=opt.csv_file.replace('.csv', '80.csv'), root=opt.root_dir, transform=transform, class_names=class_names, target_attribute=target_attribute
    )
    val_dataset = MimicCXR(
        csv_file=opt.csv_file.replace('.csv', '20.csv'), root=opt.root_dir, transform=transform, class_names=class_names, target_attribute=target_attribute
    )

    #num_samples = len(dataset)
    #indices = list(range(num_samples))
    set_seed(opt.seed)
    #np.random.shuffle(indices)

    # Define the sizes for training, validation, and test sets
    val_size = opt.val_split
    #val_split = int(np.floor(val_size * num_samples))

    #train_indices = indices[val_split:]
    #val_indices = indices[:val_split]

    train_size = len(train_dataset)
    val_size = len(val_dataset)
    #print("Dataset size:", num_samples, flush=True)
    print("Train size:", train_size, flush=True)
    print("Validation size:", val_size, flush=True)

    # Define samplers for obtaining batches from train, validation, and test sets
    #train_sampler = SubsetRandomSampler(train_indices)
    #val_sampler = SubsetRandomSampler(val_indices)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=opt.bs, num_workers=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.bs * 2, shuffle=False)

    model, criterion = set_model(opt, num_classes=1)

    decay_epochs = [opt.epochs // 3, opt.epochs * 2 // 3]
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=1e-5)

    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-2, momentum=0.9, weight_decay=1e-4)
    #optimizer = torch.optim.Adam(model.fc.parameters(), lr=opt.lr, weight_decay=opt.lr)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #    optimizer, milestones=decay_epochs, gamma=0.1
    #)


    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1)
    #scheduler = None
    logging.info(f"decay_epochs: {decay_epochs}")

    (save_path / "checkpoints").mkdir(parents=True, exist_ok=True)


    checkpoint = "./"+output_dir +"/checkpoints/last_classifier_state.pth"
    patience = 10

    print("Checking if save file", checkpoint, "exists..")
    if os.path.exists("./"+output_dir +"/checkpoints/last_classifier_state.pth"):
        print("File exists! Loading from", checkpoint)
        model, optimizer, scheduler = load_checkpoint(checkpoint, model, optimizer, scheduler)
    else:
        print("File does not exist. Starting from scratch.")

    train_model(opt, model, save_path, train_loader, val_loader, criterion, optimizer, scheduler, patience=patience, device='cuda')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info(f"Total training time: {total_time_str}")



if __name__ == "__main__":
    main()
