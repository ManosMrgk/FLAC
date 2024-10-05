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

def test_model(model, test_dataloader, criterion, device):
    class_names = ['No Finding', 'Pleural Effusion', 'Lung Opacity', 'Atelectasis']
    model.eval()
    top1 = AverageMeter()
    test_running_loss = 0.0
    test_correct_predictions = 0
    test_total_predictions = 0
    with torch.no_grad():
        with tqdm(test_dataloader, desc=f'Test calculation') as t:
            data = {col: [] for col in ['dicom_id', 'subject_id', 'study_id', 'ViewPosition', 'gender', 'insurance', 'ethnicity', 'marital_status']}
            original_labels = {class_name: [] for class_name in class_names}
            predicted_labels = {class_name + '_hat': [] for class_name in class_names}

            for inputs, labels, _ in t:
                labels_df = pd.DataFrame(inputs['dict'])


                for col in data.keys():
                    if col in labels_df.columns:
                        if col == 'ViewPosition':
                            labels_df[col] = labels_df[col].astype(str)
                        data[col].extend(labels_df[col].values)
                    else:
                        data[col].extend([0] * len(labels_df))

                inputs, labels = inputs['image'].to(device), labels.to(device)

                outputs = model(inputs)[0]
                #print(f"outputs={outputs} labels={labels}")
                test_running_loss += criterion(outputs, labels).item() * inputs.size(0)

                output_probs = outputs.cpu().detach().numpy()

                predicted_classes = (outputs > 0.5).long()
                print(f"predicted_classes:{predicted_classes} labels:{labels}")
                # true_classes = np.argmax(labels.cpu().detach().numpy(), axis=1)
                corrects = (predicted_classes == labels).float().mean() * 100
                top1.update(corrects, labels.size(0))
                #test_correct_predictions += (predicted_classes == labels).sum().item()
                test_correct_predictions += corrects
                # predicted_labels = (outputs > 0.5).float()
                # test_running_loss += criterion(outputs, labels).item() * inputs.size(0)
                # test_correct_predictions += (predicted_labels == labels).sum().item() / len(config['class_names'])
                labels_numpy = labels.detach().cpu().numpy()
                output_values = outputs.detach().cpu().numpy()
                # predicted_label_values = (outputs > 0.5).float().cpu().detach().numpy()
                # test_running_loss += criterion(outputs, labels).item() * inputs.size(0)
                # predicted_labels_tensor = torch.tensor(predicted_label_values).to(device)
                # labels_tensor = torch.tensor(labels).to(device)

                # test_correct_predictions += (predicted_labels_tensor == labels_tensor).sum().item() / len(config['class_names'])
                test_total_predictions += labels.size(0)
                for idx, class_name in enumerate(class_names):
                    original_labels[class_name].extend(labels_numpy[:, idx])
                    predicted_labels[class_name + '_hat'].extend(output_values[:, idx])
                t.set_postfix(loss=test_running_loss / test_total_predictions, accuracy=test_correct_predictions / test_total_predictions)

    print("Accuracy:", top1.avg.item())
    data.update(original_labels)
    data.update(predicted_labels)
    sanitize_data(data)
    result_df = pd.DataFrame(data)
    for col in result_df.columns:
        result_df[col] = result_df[col].apply(lambda x: x.item() if isinstance(x, torch.Tensor) else x)
    # Save to CSV
    result_df.to_csv('results20_densenet_logo_plain.csv', index=False)
    test_loss = test_running_loss / len(test_dataloader.dataset)
    test_accuracy = test_correct_predictions / test_total_predictions
    return test_loss, test_accuracy

def test(test_dataloader, criterion, model, device):
    test_loss, test_accuracy = test_model(model, test_dataloader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")


def main():
    model_path = 'results/flac-mimic_cxr_logo-logodense-lr0.001-alpha100-bs64-seed42/checkpoints/last_flac_model.pt'
    #config['model_path'] = 'models/cxr_resnet50_model.pt'
    start_time = time.time()
    print("Initialize the dataset", flush=True)
    csv_file='/home/csi22304/mimic_debiasing/data/balanced_test_small.csv'
    print("Initialize the dataset", flush=True)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    print("Using dataset csv:", csv_file)
    class_names = ['No Finding', 'Pleural Effusion', 'Lung Opacity', 'Atelectasis']
    dataset = MimicCXR(
        csv_file=csv_file, root='/home/csi22304/physionet/physionet.org/files/mimic-cxr-jpg/2.0.0/', transform=transform, class_names=class_names, testing=True,
        logo=False, gaussian_noise=False, salt_and_pepper=False,noise_intensity=35
    )
    random_state=42
    num_samples = len(dataset)
    indices = list(range(num_samples))
    np.random.seed(random_state)
    np.random.shuffle(indices)

    val_size = 0.2
    test_size = 0.2
    batch_size = 64
    val_split = int(np.floor(val_size * num_samples))
    test_split = int(np.floor(test_size * num_samples))

    test_indices = indices[val_split:val_split+test_split]

    test_size = len(test_indices)

    print("Test size:", test_size)

    test_sampler = SequentialSampler(test_indices)

    test_dataloader = DataLoader(dataset, batch_size=batch_size) #, sampler=test_sampler)

    if os.path.exists(model_path):
        model = torch.load(model_path)
        print("Model loaded from saved state:", model_path, flush=True)
    else:
        raise ValueError("Savefile not found at:", model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.BCELoss().cuda()
    print("Testing the model on the test set", flush=True)
    test(test_dataloader, criterion, model, device)

if __name__ == "__main__":
    main()
