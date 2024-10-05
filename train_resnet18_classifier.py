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
from flac import flac_loss
from datasets.mimic_cxr_jpg import MimicCXR, get_utk_face
from models.resnet import ResNet18
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


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="classifier18")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--task", type=str, default="logo")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--bs", type=int, default=32, help="batch_size")
    parser.add_argument("--lr", type=float, default=1e-2)
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
    #criterion1 = nn.CrossEntropyLoss()
    # protected_net = ResNet50()
    # if opt.task == "race":
    #     protected_attr_model = "./bias_capturing_classifiers/bcc_race.pth"
    # elif opt.task == "age":
    #     protected_attr_model = "./bias_capturing_classifiers/bcc_age.pth"
    # elif opt.task == "gender":
    #     protected_attr_model = "./bias_capturing_classifiers/bcc_gender.pth"
    # dd = load_model(protected_attr_model)
    # protected_net.load_state_dict(dd)
    # protected_net.cuda()
    return model, criterion1


def train(train_dataloader, model, criterion, optimizer, epoch, device, train_size, opt):
    print("Epoch {} running".format(epoch), flush=True)
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    with tqdm(train_dataloader, desc=f'Epoch {epoch}/{opt.epochs}') as t:
        for inputs, _, target, _ in t:
            inputs, target = inputs.to(device), target.float().to(device).view(-1, 1)
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            outputs.squeeze(1)
            loss = criterion(outputs, target)
            outputs = (torch.sigmoid(outputs) > 0.5).float()
            outputs = outputs.squeeze(1)
            print(f"Output shape: {outputs.shape}, Target shape: {target.shape}")
            #print("outputs:", outputs)
            #print("target:", target)
            #print("Unique target values:", torch.unique(target))
            #print("Model output shape:", outputs.shape)
            #print("Target shape:", target.shape)
            #print("outputs:", outputs)
            #print("target:", target)
            #print(running_loss)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

            predicted_classes = outputs
            if opt.task:
                true_classes = target
            else:
                true_classes = target.argmax(dim=1)
            print(f"Predicted: {predicted_classes} True: {true_classes}")
            correct_predictions += (predicted_classes == true_classes).sum().item()

            total_predictions += target.size(0)
            t.set_postfix(loss=running_loss / total_predictions, accuracy=correct_predictions / total_predictions)

    epoch_loss = running_loss / train_size
    train_accuracy = correct_predictions / train_size

    return model, epoch_loss, train_accuracy

def validate(val_loader, model, criterion):
    model.eval()
    correct_total = 0
    total_samples = 0
    total_val_loss = 0.0
    top1 = AverageMeter()
    attrwise_acc_meter = MultiDimAverageMeter(dims=(2, 2))

    with torch.no_grad():
        for idx, (images, _, target, _) in enumerate(val_loader):
            images, target = images.cuda(), target.float().cuda().view(-1, 1)
            bsz = target.shape[0]

            output, _ = model(images)
            output.squeeze(1)
            #preds = output.data.max(1, keepdim=True)[1].squeeze(1)
            output = (torch.sigmoid(output) > 0.5).float()
            output.squeeze(1)
            loss = criterion(output, target)
            total_val_loss += loss.item() * bsz

            (acc1,) = accuracy(output, target, topk=(1,))
            top1.update(acc1[0], bsz)

            corrects = (output == target).long()
            correct_total += output.eq(target).sum().item()
            total_samples += bsz

            #print("output: ", output)
            #print("preds: ", preds)
            #print("target: ", target)
            #print("corrects: ", corrects)
            if target.dim() == 1:
                target = target.unsqueeze(1)
            corrects_float = corrects.float()
            #idx_tensor = torch.cat((target, corrects_float.unsqueeze(1)), dim=1).long()

            #attrwise_acc_meter.add(corrects.cpu(), idx_tensor.cpu())

    overall_accuracy = 100.0 * correct_total / total_samples
    val_loss = total_val_loss / total_samples
    return overall_accuracy, val_loss
    #return top1.avg, attrwise_acc_meter.get_mean()

def save_checkpoint(model, optimizer, lr_scheduler, best_accs, best_epochs, best_stats, path):
    path = str(path)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        'best_accs': best_accs,
        'best_epochs': best_epochs,
        'best_stats': best_stats
    }, path+"_state.pth")
    print("Saving model at:", path+".pth")
    torch.save(model, path+".pth")

def load_checkpoint(path, model, optimizer, lr_scheduler):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
    best_accs = checkpoint['best_accs']
    best_epochs = checkpoint['best_epochs']
    best_stats = checkpoint['best_stats']
    return model, optimizer, lr_scheduler, best_accs, best_epochs, best_stats

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
    ])
    print("Using dataset csv:", opt.csv_file)
    class_names = ['No Finding', 'Pleural Effusion', 'Lung Opacity', 'Atelectasis']
    target_attribute = opt.task
    dataset = MimicCXR(
        csv_file=opt.csv_file, root=opt.root_dir, transform=transform, class_names=class_names, target_attribute=target_attribute
    )

    num_samples = len(dataset)
    indices = list(range(num_samples))
    set_seed(opt.seed)
    np.random.shuffle(indices)

    # Define the sizes for training, validation, and test sets
    val_size = opt.val_split
    val_split = int(np.floor(val_size * num_samples))

    train_indices = indices[val_split:]
    val_indices = indices[:val_split]

    train_size = len(train_indices)
    val_size = len(val_indices)
    print("Dataset size:", num_samples, flush=True)
    print("Train size:", train_size, flush=True)
    print("Validation size:", val_size, flush=True)

    # Define samplers for obtaining batches from train, validation, and test sets
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    # Create data loaders
    train_loader = DataLoader(dataset, batch_size=opt.bs, sampler=train_sampler, num_workers=2)
    val_loader = DataLoader(dataset, batch_size=opt.bs * 2, sampler=val_sampler)

    model, criterion = set_model(opt, num_classes=1)

    decay_epochs = [opt.epochs // 3, opt.epochs * 2 // 3]

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=decay_epochs, gamma=0.1
    )


    logging.info(f"decay_epochs: {decay_epochs}")

    (save_path / "checkpoints").mkdir(parents=True, exist_ok=True)

    best_accs = {"val": 0}
    best_epochs = {"val": 0}
    best_stats = {}
    checkpoint = "./"+output_dir +"/checkpoints/last_classifier_state.pth"
    patience = 10
    best_loss = float('inf')
    patience_counter = 0
    print("Checking if save file", checkpoint, "exists..")
    if os.path.exists("./"+output_dir +"/checkpoints/last_classifier_state.pth"):
        print("File exists! Loading from", checkpoint)
        model, optimizer, scheduler, best_accs, best_epochs, best_stats = load_checkpoint(checkpoint, model, optimizer, scheduler)
    else:
        print("File does not exist. Starting from scratch.")

    for epoch in range(1, opt.epochs + 1):
        logging.info(
            f"[{epoch} / {opt.epochs}] Learning rate: {scheduler.get_last_lr()[0]}"
        )
        model, epoch_loss, train_accuracy = train(train_loader, model, criterion, optimizer, epoch, device, train_size, opt)

        logging.info(
            f"[{epoch} / {opt.epochs}] Loss: {epoch_loss}  Train Acc: {train_accuracy} Patience buildup: {patience_counter}"
        )

        scheduler.step()

        stats = pretty_dict(epoch=epoch)


        accs, val_loss = validate(val_loader, model, criterion)
        logging.info(f"val_loss: {val_loss}")
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping at epoch:", epoch)
            break

        stats[f"val/acc"] = accs
        #stats[f"val/acc_unbiased"] = torch.mean(valid_attrwise_accs).item() * 100
        #eye_tsr = torch.eye(2)
        #stats[f"val/acc_skew"] = (
        #    valid_attrwise_accs[eye_tsr == 0.0].mean().item() * 100
        #)
        #stats[f"val/acc_align"] = (
        #    valid_attrwise_accs[eye_tsr > 0.0].mean().item() * 100
        #)

        logging.info(f"[{epoch} / {opt.epochs}] {stats}")
        print('val', best_accs, 'best_stats:', best_stats)
        print("best_accs[val]:", best_accs['val'])
        if stats[f"val/acc"] > best_accs['val']:
            best_accs['val'] = stats[f"val/acc"]
            best_epochs['val'] = epoch
            best_stats['val'] = pretty_dict(
                **{f"best_val_{k}": v for k, v in stats.items()}
            )

            save_file = save_path / "checkpoints" / f"best_val_{opt.task}_classifier"
            save_checkpoint(model, optimizer, scheduler, best_accs, best_epochs, best_stats, save_file)

            logging.info(
                f"[{epoch} / {opt.epochs}] best val accuracy: {best_accs['val']:.3f} at epoch {best_epochs['val']} \n best_stats: {best_stats.get('val', '-')}"
            )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info(f"Total training time: {total_time_str}")

    save_file = save_path / "checkpoints" / f"last_classifier"
    # save_model(model, optimizer, opt, opt.epochs, save_file)
    save_checkpoint(model, optimizer, scheduler, best_accs, best_epochs, best_stats, save_file)
    protected_attr_model = None
    if opt.task == "race":
        protected_attr_model = "./bias_capturing_classifiers/bcc_race.pth"
    elif opt.task == "age":
        protected_attr_model = "./bias_capturing_classifiers/bcc_age.pth"
    elif opt.task == "gender":
        protected_attr_model = "./bias_capturing_classifiers/bcc_gender.pth"
    elif opt.task == "logo":
        protected_attr_model = "./bias_capturing_classifiers/bcc_logo.pth"
    torch.save(model, protected_attr_model)

if __name__ == "__main__":
    main()
