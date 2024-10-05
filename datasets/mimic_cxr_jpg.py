import logging
import os
import pickle
from pathlib import Path
import pandas as pd
import PIL
import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm
from datasets.utils import TwoCropTransform, get_confusion_matrix
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
from PIL import Image


class MimicCXR:
    def __init__(
        self, csv_file, root, transform, class_names=['Enlarged Cardiomediastinum', 'No Finding'], testing=False, target_attribute=None, chunk_size=15000, logo=False, gaussian_noise=False, salt_and_pepper=False, brightness_bands=False, noise_intensity=35, **kwargs
    ):
        self.root = Path(root)
        self.testing = testing
        self.target_attribute = target_attribute
        self.chunk_size = chunk_size
        # Load metadata CSV
        self.data_frame = pd.read_csv(csv_file)
        #self.data_frame = self.data_frame[:1000]
        # Filter dataframe to include only rows with selected classes
        self.data_frame = self.data_frame[self.data_frame[class_names].sum(axis=1) > 0]

        # Remove rows that have -1 as any of the values of the selected classes
        self.data_frame = self.data_frame[(self.data_frame[class_names] != -1).all(axis=1)]

        # Marking 90% of Pleural Effusion images for adding logos or noise
        if (logo or gaussian_noise or salt_and_pepper or brightness_bands):
            print(f"Adding device bias: logo: {logo}, gaussian_noise: {gaussian_noise}, salt_and_pepper: {salt_and_pepper}, brightness_bands: {brightness_bands}")
            self.mark_selected_images()

        if self.target_attribute in ['logo', 'gaussian_noise', 'salt_and_pepper', 'brightness_bands']:
            print(f"Adding target attribute '{self.target_attribute}' to half the images")
            self.mark_selected_images_device_bias()
            #print("Logo:", self.target_attribute == 'logo')
            logo = self.target_attribute == 'logo'
            gaussian_noise = self.target_attribute == 'gaussian_noise'
            salt_and_pepper = self.target_attribute == 'salt_and_pepper'
            brightness_bands = self.target_attribute == 'brightness_bands'

        # Get class indices after filtering
        self.class_indices = [self.data_frame.columns.get_loc(class_name) for class_name in class_names]

        num_files = len(self.data_frame)
        num_train = int(num_files * 0.8)

        self.transform = transform
        self.resize_transform = transforms.Compose([
            transforms.Resize((224, 224))
        ])
        self.class_names = class_names

        self.images = []
        self.labels = []
        images_file = os.path.splitext(csv_file)[0]+'_images.pt'
        labels_file = os.path.splitext(csv_file)[0]+'_labels.pkl'
        data_dir = os.path.dirname(csv_file)
        image_file_exists = len([name for name in os.listdir(data_dir) if name.startswith(os.path.splitext(os.path.basename(csv_file))[0]+'_images')]) > 0
        print(data_dir, os.path.splitext(os.path.basename(csv_file))[0]+'_images', image_file_exists)
        if os.path.exists(labels_file) and image_file_exists:
            # Load preprocessed data
            self.images = []
            for chunk_file in sorted(os.listdir(data_dir)):
                if chunk_file.startswith(os.path.splitext(os.path.basename(csv_file))[0]+'_images'):
                    print("Loading file:", chunk_file, flush=True)
                    images_chunk = torch.load(os.path.join(data_dir, chunk_file), map_location='cpu')
                    self.images.append(images_chunk)
            #self.images = torch.cat(self.images, dim=0)
            self.images = torch.utils.data.ConcatDataset(self.images)
            with open(labels_file, 'rb') as f:
                self.data_frame = pickle.load(f)
                #self.data_frame.to_csv('output.csv', index=False) 
                self.labels = self.data_frame[self.class_names].values.astype(np.float32)
            print(f"Loaded preprocessed data for {csv_file}", flush=True)
        else:
            has_marks = 'mark' in self.data_frame.columns
            #print("has_marks:", has_marks)
            #print(self.data_frame.columns)
            #print("Logo:", logo)
            if has_marks and logo:
                icon_image_path = '/home/csi22304/mimic_debiasing/hua_logo.png'
                print("Loading logo from:", icon_image_path)
                icon_image = Image.open(icon_image_path)
                icon_image = icon_image.convert("RGBA")
                icon_size = (20, 20)
                icon_image = icon_image.resize(icon_size)
            for idx in tqdm(range(len(self.data_frame)), desc="Processing Images"):
                labels = self.data_frame.iloc[idx, self.class_indices].values.astype(np.float32)
                self.labels.append(labels)
                #if idx < 75000:
                #   continue
                img_path = os.path.join(self.root, self.data_frame.iloc[idx]['path'])
                image = Image.open(img_path).convert('RGB')
                if has_marks:
                    if self.data_frame.iloc[idx]['mark'] == 1:
                        image = self.resize_transform(image)
                        if logo:
                            image = self.add_logo_to_image(image, icon_image)
                            #self.save_image(image, f"{os.path.splitext(csv_file)[0]}_imagelogo.png")
                            #exit()
                        if gaussian_noise:
                            image = self.add_gaussian_noise_to_image(image, std=noise_intensity)
                        if salt_and_pepper:
                            image = self.add_salt_and_pepper_to_image(image, noise_intensity=noise_intensity/1000)
                        if brightness_bands:
                            image = self.add_horizontal_brightness_bands(image)
                        tensor_transform = transforms.ToTensor()
                        image = tensor_transform(image)
                        norm_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        image = norm_transform(image)
                    elif self.transform:
                        image = self.transform(image)
                else:
                    if self.transform:
                        image = self.transform(image)
                self.images.append(image)


                if (idx + 1) % self.chunk_size == 0:
                   chunk_idx = (idx + 1) // self.chunk_size
                   images_chunk = torch.stack(self.images)
                   torch.save(images_chunk, f'{os.path.splitext(csv_file)[0]}_images_chunk_{chunk_idx:04d}.pt')
                   print(f"Saved {chunk_idx * self.chunk_size} images incrementally in chunk number: {chunk_idx:04d}.")
                   self.images = []
                   del images_chunk
                   torch.cuda.empty_cache()
            # Save any remaining data
            if len(self.data_frame) % self.chunk_size != 0:
                chunk_idx = len(self.data_frame) // self.chunk_size + 1
                images_chunk = torch.stack(self.images[-(len(self.images) % self.chunk_size):])
                torch.save(images_chunk, f'{os.path.splitext(csv_file)[0]}_images_chunk_{chunk_idx:04d}.pt')
                print(f"Saved remaining images incrementally in chunk number: {chunk_idx:04d}.")
            self.images = []
            with open(labels_file, 'wb') as f:
                pickle.dump(self.data_frame, f)
            del images_chunk
            torch.cuda.empty_cache()
            filename_with_ext = os.path.basename(csv_file)
            filename_without_ext = os.path.splitext(filename_with_ext)[0]
            for chunk_file in sorted(os.listdir(data_dir)):
                #if chunk_file.startswith(os.path.splitext(csv_file)[0]+'_images'):
                if chunk_file.startswith(filename_without_ext+'_images'):
                    print("Loading file:", chunk_file, flush=True)
                    images_chunk = torch.load(os.path.join(data_dir, chunk_file))
                    self.images.append(images_chunk)
            self.images = torch.utils.data.ConcatDataset(self.images)
            #self.images = torch.cat(self.images, dim=0)
            # self.images = torch.stack(self.images)
            self.labels = np.array(self.labels)

            # Save the preprocessed data
            # torch.save(self.images, images_file)
            with open(labels_file, 'wb') as f:
                pickle.dump(self.data_frame, f)
            print(f"Saved preprocessed data to {labels_file}", flush=True)
            exit()

    def add_gaussian_noise_to_image(self, image, std=35, mean=0):
        image_array = np.array(image)
        gaussian_noise = np.random.normal(mean, std, image_array.shape)
        noisy_image_array = image_array + gaussian_noise
        noisy_image_array = np.clip(noisy_image_array, 0, 255).astype(np.uint8)
        noisy_image = Image.fromarray(noisy_image_array)

        return noisy_image

    def save_image(self, img, save_path):
        img = img.convert("RGBA")
        img.save(save_path, format='PNG')

    def add_salt_and_pepper_to_image(self, image, noise_intensity):
        image_array = np.array(image)
        total_pixels = image_array.shape[0] * image_array.shape[1]

        # Adding salt noise
        num_salt = int(np.ceil(noise_intensity * total_pixels))
        coords = [np.random.randint(0, i - 1, num_salt) for i in image_array.shape[:2]]
        image_array[coords[0], coords[1], :] = 255

        # Adding pepper noise
        num_pepper = int(np.ceil(noise_intensity * total_pixels))
        coords = [np.random.randint(0, i - 1, num_pepper) for i in image_array.shape[:2]]
        image_array[coords[0], coords[1], :] = 0

        noisy_image = Image.fromarray(image_array)

        return noisy_image

    def add_logo_to_image(self, img, logo, padding=5):
        main_width, main_height = img.size
        icon_width, icon_height = logo.size
        position = (main_width - icon_width - padding, main_height - icon_height - padding)
        img.paste(logo, position, logo)

        return img
    
    def add_horizontal_brightness_bands(self, image, num_bands=30, mean_band_thickness=10, low_intensity_factor=0.9, high_intensity_factor=1.1):
        image_array = np.array(image)
        height, width, _ = image_array.shape

        start_y = int(height * 4 / 5)  # Starting at 4/5 of the image (lower 1/5)
        end_y = height

        # Apply horizontal brightness bands
        for i, y_position in enumerate(np.linspace(start_y, end_y, num_bands)):
            y_position = int(y_position)
            band_thickness = max(1, int(np.random.normal(mean_band_thickness, mean_band_thickness * 0.2)))
            # print(band_thickness)

            if y_position + band_thickness > height:
                band_thickness = height - y_position

            if i % 2 == 0:  # Even num: lower brightness
                image_array[y_position:y_position + band_thickness, :, :] = (image_array[y_position:y_position + band_thickness, :, :] * low_intensity_factor).clip(0, 255)
            else:  # Odd num: higher brightness
                image_array[y_position:y_position + band_thickness, :, :] = (image_array[y_position:y_position + band_thickness, :, :] * high_intensity_factor).clip(0, 255)

        return Image.fromarray(image_array)

    def mark_selected_images_device_bias(self, ratio=0.5):
        self.data_frame['mark'] = 0
        num_images_with_mark = int(ratio * len(self.data_frame))
        selected_indices = self.data_frame.sample(n=num_images_with_mark, replace=False).index
        self.data_frame.loc[selected_indices, 'mark'] = 1

    def mark_selected_images(self, selected_class='Pleural Effusion', ratio=0.9):
        if not (0 <= ratio <= 1):
            raise ValueError("Ratio must be between 0 and 1.")
        # adding a new column to self.data_frame that indicates if the logo or noise will be added or not with 1 and 0 based on the ratio specified
        self.data_frame['mark'] = 0
        class_indices = self.data_frame.index[self.data_frame[selected_class] == 1].tolist()

        class_mask = self.data_frame[selected_class] == 1
        num_images_with_mark = int(ratio * len(class_indices))
        selected_indices = class_indices[:num_images_with_mark]
        self.data_frame.loc[selected_indices, 'mark'] = 1

    def __getitem__(self, idx):
        image = self.images[idx]

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
            elif self.target_attribute in ['logo', 'gaussian_noise', 'salt_and_pepper', 'brightness_bands']:
                mark = self.data_frame.iloc[idx]['mark']
                return image, torch.tensor(labels), torch.tensor(mark), idx

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

