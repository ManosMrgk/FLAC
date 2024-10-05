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
from flac import flac_loss
from datasets.mimic_cxr_jpg import MimicCXR, get_utk_face
from models.resnet import ResNet50, ResNet18
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


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="logo")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--task", type=str, default="logo")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--bs", type=int, default=64, help="batch_size")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=100)
    parser.add_argument("--csv_file", type=str, default='/home/csi22304/mimic_debiasing/data/half_meta_data_filtered.csv')
    parser.add_argument("--root_dir", type=str, default='/home/csi22304/physionet/physionet.org/files/mimic-cxr-jpg/2.0.0/')
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--test_split", type=float, default=0.2)
    parser.add_argument("--logo", default=False, action="store_true")
    parser.add_argument("--gaussian_noise", default=False, action="store_true")
    parser.add_argument("--salt_and_pepper", default=False, action="store_true")
    parser.add_argument("--brightness_bands", default=False, action="store_true")
    parser.add_argument("--noise_intensity", type=int, default=35)
    opt = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)

    return opt


def set_model(opt, num_classes=2):
    model = ResNet18(num_classes=num_classes).cuda()
    print(model)
    if True:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.extractor[6:].parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
    criterion1 = nn.CrossEntropyLoss()
    protected_net = ResNet18(num_classes=1)
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
    protected_net = torch.load(protected_attr_model)
    protected_net.cuda()
    return model, criterion1, protected_net


def train(train_loader, model, criterion, optimizer, protected_net, opt):
    model.train()
    avg_loss = AverageMeter()
    avg_clloss = AverageMeter()
    avg_miloss = AverageMeter()
    total_b_pred = 0
    total = 0
    train_iter = iter(train_loader)
    total_steps = len(train_iter)
    with tqdm(train_loader, desc=f'Training') as t:
        for images, labels, biases, _ in t:
            bsz = labels.shape[0]
            labels, biases = labels.cuda(), biases.cuda()
            labels = torch.argmax(labels, dim=1)
            images = images.cuda()
            logits, features = model(images)
            with torch.no_grad():
                pr_l, pr_feat = protected_net(images)
                predicted_race = pr_l.argmin(dim=1, keepdim=True)

            predicted_race = predicted_race.T
            # print(f'pr_feat shape: {pr_feat.shape}')
            # print(f'features shape: {features.shape}')
            # print(f'labels shape: {labels.shape}')
            loss_mi_div = opt.alpha * (flac_loss(pr_feat, features, labels))
            loss_cl = 0.01 * criterion(logits, labels)
            loss = loss_cl + loss_mi_div

            avg_loss.update(loss.item(), bsz)
            avg_clloss.update(loss_cl.item(), bsz)
            avg_miloss.update(loss_mi_div.item(), bsz)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_b_pred += predicted_race.eq(biases.view_as(predicted_race)).sum().item()
            total += bsz
            t.set_postfix(loss=avg_loss.avg)
    return avg_loss.avg, avg_clloss.avg, avg_miloss.avg


def validate(val_loader, model, criterion):
    model.eval()

    top1 = AverageMeter()
    attrwise_acc_meter = MultiDimAverageMeter(dims=(4, 2))
    total_val_loss = 0.0
    with torch.no_grad():
        for idx, (images, labels, biases, _) in enumerate(val_loader):
            images, labels, biases = images.cuda(), labels.cuda(), biases.cuda()
            bsz = labels.shape[0]

            output, _ = model(images)
            loss = criterion(output, labels)
            total_val_loss += loss.item() * bsz
            preds = output.data.max(1, keepdim=True)[1].squeeze(1)
            #print(f'Output shape: {output.shape}, Labels shape: {labels.shape}')

            if labels.size(1) > 1:
                labels = torch.argmax(labels, dim=1)

            (acc1,) = accuracy(output, labels, topk=(1,))
            top1.update(acc1[0], bsz)

            corrects = (preds == labels).long()
            attrwise_acc_meter.add(
                corrects.cpu(), torch.stack([labels.cpu(), biases.cpu()], dim=1)
            )

    return total_val_loss, top1.avg, attrwise_acc_meter.get_mean()

def save_checkpoint(model, optimizer, lr_scheduler, best_accs, best_epochs, best_stats, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        'best_accs': best_accs,
        'best_epochs': best_epochs,
        'best_stats': best_stats
    }, path)

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
        logo=opt.logo, gaussian_noise=opt.gaussian_noise, salt_and_pepper=opt.salt_and_pepper, brightness_bands=opt.brightness_bands, noise_intensity=opt.noise_intensity
    )

    num_samples = len(dataset)
    indices = list(range(num_samples))
    set_seed(opt.seed)
    np.random.shuffle(indices)

    # Define the sizes for training, validation, and test sets
    val_size = opt.val_split
    test_size = opt.test_split
    val_split = int(np.floor(val_size * num_samples))
    test_split = int(np.floor(test_size * num_samples))

    #train_indices = indices[val_split + test_split:]
    train_indices = indices[val_split:]
    val_indices = indices[:val_split]
    test_indices = indices[val_split:val_split+test_split]

    train_size = len(train_indices)
    val_size = len(val_indices)
    test_size = len(test_indices)
    print("Dataset size:", num_samples, flush=True)
    print("Train size:", train_size, flush=True)
    print("Validation size:", val_size, flush=True)
    print("Test size:", test_size, flush=True)

    # Define samplers for obtaining batches from train, validation, and test sets
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SequentialSampler(val_indices)
    test_sampler = SequentialSampler(test_indices)

    # Create data loaders
    train_loader = DataLoader(dataset, batch_size=opt.bs, sampler=train_sampler, num_workers=2)
    val_loaders = {}
    val_loaders["valid"] = DataLoader(dataset, batch_size=opt.bs * 2, sampler=val_sampler)
    val_loaders["test"] = DataLoader(dataset, batch_size=opt.bs * 2, sampler=test_sampler)

    model, criterion, protected_net = set_model(opt, num_classes=len(class_names))

    decay_epochs = [opt.epochs // 3, opt.epochs * 2 // 3]

    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=1e-5)
    # optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #     optimizer, milestones=decay_epochs, gamma=0.1
    # )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)


    logging.info(f"decay_epochs: {decay_epochs}")

    (save_path / "checkpoints").mkdir(parents=True, exist_ok=True)

    best_accs = {"valid": 0, "test": 0}
    best_epochs = {"valid": 0, "test": 0}
    best_stats = {}
    if os.path.exists(save_path / "checkpoints" / f"last.pth"):
        print("Loading from savefile")
        model, optimizer, scheduler, best_accs, best_epochs, best_stats = load_checkpoint(save_path / "checkpoints" / f"last.pth", model, optimizer, scheduler)
        torch.save(model, save_path / 'checkpoints' / 'last_flac_model.pt')

    if True:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.extractor[6:].parameters():
            param.requires_grad = True
        for param in model.fc.parameters():
            param.requires_grad = True

    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    for epoch in range(1, opt.epochs + 1):
        if epoch > opt.epochs // 4:
            for param in model.extractor[3:].parameters():
                param.requires_grad = True
        validation_loss = None
        logging.info(
            f"[{epoch} / {opt.epochs}] Learning rate: {optimizer.param_groups[0]['lr']}" # {scheduler.get_last_lr()[0]}"
        )
        loss, cllossp, milossp = train(
            train_loader, model, criterion, optimizer, protected_net, opt
        )
        logging.info(
            f"[{epoch} / {opt.epochs}] Loss: {loss}  Loss CE: {cllossp}  Loss MI: {milossp}"
        )

        # scheduler.step()
        
        stats = pretty_dict(epoch=epoch)
        for key, val_loader in val_loaders.items():
            val_loss, accs, valid_attrwise_accs = validate(val_loader, model, criterion)
            if key == 'valid':
                validation_loss = val_loss
            stats[f"{key}/acc"] = accs.item()
            stats[f"{key}/acc_unbiased"] = torch.mean(valid_attrwise_accs).item() * 100
            eye_tsr = torch.eye(4)[:,:2]
            stats[f"{key}/acc_skew"] = (
                valid_attrwise_accs[eye_tsr == 0.0].mean().item() * 100
            )
            stats[f"{key}/acc_align"] = (
                valid_attrwise_accs[eye_tsr > 0.0].mean().item() * 100
            )

        logging.info(f"[{epoch} / {opt.epochs}] Val_loss: {validation_loss} {valid_attrwise_accs} {stats}")
        for tag in val_loaders.keys():
            if stats[f"{tag}/acc_unbiased"] > best_accs[tag]:
                best_accs[tag] = stats[f"{tag}/acc_unbiased"]
                best_epochs[tag] = epoch
                best_stats[tag] = pretty_dict(
                    **{f"best_{tag}_{k}": v for k, v in stats.items()}
                )

                save_file = save_path / "checkpoints" / f"best_{tag}.pth"
                save_checkpoint(model, optimizer, scheduler, best_accs, best_epochs, best_stats, save_file)
            logging.info(
                f"[{epoch} / {opt.epochs}] best {tag} accuracy: {best_accs[tag]:.3f} at epoch {best_epochs[tag]} \n best_stats: {best_stats.get(tag, '-')}"
            )
        scheduler.step(validation_loss)
        if validation_loss < best_val_loss:
            logging.info(f"Validation loss improved from {best_val_loss:.4f} to {validation_loss:.4f}. Saving model...")
            best_val_loss = validation_loss  
            patience_counter = 0     
            torch.save(model, save_path / 'checkpoints' / 'best_flac_model.pt')
        else:
            patience_counter += 1
            logging.info(f"Patience counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                logging.info("Early stopping triggered.")
                break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info(f"Total training time: {total_time_str}")
    torch.save(model, save_path / 'checkpoints' / 'last_flac_model.pt')
    save_file = save_path / "checkpoints" / f"last.pth"
    # save_model(model, optimizer, opt, opt.epochs, save_file)
    save_checkpoint(model, optimizer, scheduler, best_accs, best_epochs, best_stats, save_file)

if __name__ == "__main__":
    main()

