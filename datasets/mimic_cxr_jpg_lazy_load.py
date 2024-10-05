import logging
import os
import pickle
from pathlib import Path
import pandas as pd
import PIL
import numpy as np
import torch
import torch.utils.data
from datasets.utils import TwoCropTransform, get_confusion_matrix
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
from PIL import Image


class MimicCXR:
    def __init__(
        self, csv_file, root, transform, class_names=['Enlarged Cardiomediastinum', 'No Finding'], testing=False, target_attribute=None, **kwargs
    ):
        self.root = Path(root)
        self.testing = testing
        self.target_attribute = target_attribute
        # Load metadata CSV
        self.data_frame = pd.read_csv(csv_file)
        #self.data_frame = self.data_frame[:1000]
        # Filter dataframe to include only rows with selected classes
        self.data_frame = self.data_frame[self.data_frame[class_names].sum(axis=1) > 0]

        # Remove rows that have -1 as any of the values of the selected classes
        self.data_frame = self.data_frame[(self.data_frame[class_names] != -1).all(axis=1)]

        # Get class indices after filtering
        self.class_indices = [self.data_frame.columns.get_loc(class_name) for class_name in class_names]
        
        num_files = len(self.data_frame)
        num_train = int(num_files * 0.8)

        self.transform = transform
        self.class_names = class_names


    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.data_frame.iloc[idx]['path'])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        # labels[np.isnan(labels)] = 0
        labels = self.data_frame.iloc[idx, self.class_indices].values.astype(np.float32)
        nan_mask = np.isnan(labels)
        labels[nan_mask] = 0

        if self.target_attribute:
            if self.target_attribute == 'gender':
                gender = self.data_frame.iloc[idx]['gender']
                gender_label = 1 if gender == 'M' else 0
                return image, torch.tensor(labels), torch.tensor(gender_label), idx
            elif self.target_attribute == 'age':
                age = self.data_frame.iloc[idx]['age']
                age_label = 1 if age > 40 else 0
                return image, torch.tensor(labels), torch.tensor(age_label), idx


        if self.testing:
            labels_dict = self.data_frame.iloc[idx].to_dict()  
            for key, value in labels_dict.items():
                if isinstance(value, (int, float)):
                    labels_dict[key] = str(value)
            return {'image':image, 'dict':labels_dict}, labels, idx


        return image, torch.tensor(labels), idx

        # Was: return X(the image), target, bias?, idx

    def __len__(self):
        return len(self.data_frame)


def get_utk_face(
    root,
    batch_size,
    split,
    bias_attr="race",
    bias_rate=0.9,
    num_workers=8,
    aug=False,
    image_size=64,
    two_crop=False,
    ratio=0,
    given_y=True,
):
    logging.info(
        f"get_utk_face - split: {split}, aug: {aug}, given_y: {given_y}, ratio: {ratio}"
    )
    size_dict = {64: 72, 128: 144, 224: 256}
    load_size = size_dict[image_size]
    train = split == "train"

    if train:
        if aug:
            transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(size=image_size, scale=(0.2, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply(
                        [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
                    ),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )

    else:
        transform = transforms.Compose(
            [
                transforms.Resize(load_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

    if two_crop:
        transform = TwoCropTransform(transform)

    dataset = MimicCXR(
        root, transform=transform
    )

    def clip_max_ratio(score):
        upper_bd = score.min() * ratio
        return np.clip(score, None, upper_bd)

    if ratio != 0:
        if given_y:
            weights = [
                1 / dataset.confusion_matrix_by[c, b]
                for c, b in zip(dataset.targets, dataset.bias_targets)
            ]
        else:
            weights = [
                1 / dataset.confusion_matrix[b, c]
                for c, b in zip(dataset.targets, dataset.bias_targets)
            ]
        if ratio > 0:
            weights = clip_max_ratio(np.array(weights))
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    else:
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=two_crop,
    )

    return dataloader
