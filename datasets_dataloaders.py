import json
import multiprocessing
import os
import random
import sys

from PIL import Image
from rand_augment import RandAugment
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

"""

(A) DatasetsUtils: 
    - generate dataset according to the input image path and labels list; 

(A) DatasetsDataloaders - get_dataset(image_paths, labels, transform, is_train=False, N=0, M=0):
    - get the dataset with paths and labels; 
    - use the is_train to determine where the dataset is for training or inference; 

"""


class DatasetsDataloadersUtils:
    @staticmethod
    def map_classnames_to_classindex(name_labels_list):
        class_names = sorted(list(set(name_labels_list)))
        class_to_idx_file = os.path.join("./work_dirs", "class_to_idx.json")

        if os.path.isfile(class_to_idx_file):
            print(f"The JSON file '{class_to_idx_file}' exists.")
            with open(class_to_idx_file, "r") as json_file:
                class_to_idx = json.load(json_file)

            # Check if class_names match the keys in class_to_idx
            if set(class_names) != set(class_to_idx.keys()):
                print("Error: Class names do not match the keys in class_to_idx.")
                print("Exiting the code.")
                sys.exit(1)

            label_indices = [class_to_idx[label] for label in name_labels_list]
        else:
            print(f"The JSON file '{class_to_idx_file}' does not exist.")

            class_to_idx = {
                class_name: idx for idx, class_name in enumerate(class_names)
            }
            label_indices = [class_to_idx[label] for label in name_labels_list]

            with open(class_to_idx_file, "w") as json_file:
                json.dump(class_to_idx, json_file)

        return label_indices, class_to_idx

    @staticmethod
    def save_list_to_txtfile(my_list, file_path):
        with open(file_path, "w") as file:
            for item in my_list:
                file.write(str(item) + "\n")
        print(f"List saved to {file_path}")
        return file_path

    @staticmethod
    def read_list_from_txtfile(file_path):
        my_list = []
        with open(file_path, "r") as file:
            for line in file:
                my_list.append(line.strip())
        print("List read from the file:", len(my_list))
        return my_list

    @staticmethod
    def get_images_in_folder(folder_path, num_samples=None):
        image_extensions = [
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".bmp",
            ".tiff",
            ".webp",
        ]
        image_files = []
        for filename in os.listdir(folder_path):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(folder_path, filename))

        if num_samples is not None:
            image_files = random.sample(image_files, num_samples)

        return image_files

    @staticmethod
    def get_image_paths_and_labels_from_root(
        root_folder, num_samples=None, is_train=False
    ):
        class_list = [
            d
            for d in os.listdir(root_folder)
            if os.path.isdir(os.path.join(root_folder, d))
        ]
        print(f"==>> class_list: {len(class_list)}")

        image_list = []
        label_list = []
        for class_id in class_list:
            class_folder_path = os.path.join(root_folder, class_id)
            image_list_per_class = DatasetsDataloadersUtils.get_images_in_folder(
                class_folder_path, num_samples=num_samples
            )
            image_list.extend(image_list_per_class)
            lables_per_class = [class_id] * len(image_list_per_class)
            label_list.extend(lables_per_class)

        work_dirs = "./work_dirs"
        os.makedirs(work_dirs, exist_ok=True)
        if is_train:
            images_path_txtfile = os.path.join(work_dirs, "train_images.txt")
            labels_path_txtfile = os.path.join(work_dirs, "train_labels.txt")
        else:
            images_path_txtfile = os.path.join(work_dirs, "val_images.txt")
            labels_path_txtfile = os.path.join(work_dirs, "val_labels.txt")
        DatasetsDataloadersUtils.save_list_to_txtfile(image_list, images_path_txtfile)
        DatasetsDataloadersUtils.save_list_to_txtfile(label_list, labels_path_txtfile)


class DatasetsUtils:
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class DatasetsDataloaders(Dataset):
    @staticmethod
    def get_dataset(image_paths, labels, transform=None, is_train=False, N=0, M=0):
        if transform is None:
            if is_train:
                transform = transforms.Compose(
                    [
                        transforms.RandomResizedCrop(224),  # Resize and crop to 224x224
                        transforms.RandomHorizontalFlip(),  # Randomly flip horizontally
                        transforms.ToTensor(),  # Convert to a PyTorch tensor
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),  # Normalize using ImageNet statistics
                    ]
                )
                transform.transforms.insert(0, RandAugment(N, M))
            else:
                transform = transforms.Compose(
                    [
                        transforms.Resize(256),  # Resize to 256x256
                        transforms.CenterCrop(224),  # Crop the center to 224x224
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]
                )

        dataset = DatasetsUtils(
            image_paths=image_paths, labels=labels, transform=transform
        )
        return dataset

    @staticmethod
    def get_dataloader(dataset, batch_size, shuffle):
        num_cpus = multiprocessing.cpu_count()
        num_workers = min(4 * num_cpus, 32)
        pin_memory = True
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        return dataloader


if __name__ == "__main__":
    val_image_root_folder = "/home/xma24/data/imagenet/val/"
    DatasetsDataloadersUtils.get_image_paths_and_labels_from_root(
        val_image_root_folder, num_samples=10, is_train=False
    )

    train_image_root_folder = "/home/xma24/data/imagenet/train/"
    DatasetsDataloadersUtils.get_image_paths_and_labels_from_root(
        train_image_root_folder, num_samples=30, is_train=True
    )

    val_images_txtfile = os.path.join("./work_dirs/val_images.txt")
    val_labels_txtfile = os.path.join("./work_dirs/val_labels.txt")
    val_image_paths = DatasetsDataloadersUtils.read_list_from_txtfile(
        val_images_txtfile
    )
    val_labels = DatasetsDataloadersUtils.read_list_from_txtfile(val_labels_txtfile)
    (
        val_label_indices,
        class_to_idx,
    ) = DatasetsDataloadersUtils.map_classnames_to_classindex(val_labels)

    # val_transform = transforms.Compose(
    #     [
    #         transforms.Resize(256),  # Resize to 256x256
    #         transforms.CenterCrop(224),  # Crop the center to 224x224
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ]
    # )

    val_dataset = DatasetsDataloaders.get_dataset(
        image_paths=val_image_paths,
        labels=val_label_indices,
        transform=None,
        is_train=False,
        N=None,
        M=None,
    )

    val_dataloader = DatasetsDataloaders.get_dataloader(
        val_dataset, batch_size=32, shuffle=False
    )
    print(f"==>> val_dataloader: {len(val_dataloader)}")

    train_images_txtfile = os.path.join("./work_dirs/train_images.txt")
    train_labels_txtfile = os.path.join("./work_dirs/train_labels.txt")
    train_image_paths = DatasetsDataloadersUtils.read_list_from_txtfile(
        train_images_txtfile
    )
    train_labels = DatasetsDataloadersUtils.read_list_from_txtfile(train_labels_txtfile)
    (
        train_label_indices,
        class_to_idx,
    ) = DatasetsDataloadersUtils.map_classnames_to_classindex(train_labels)

    # train_transform = transforms.Compose(
    #     [
    #         transforms.RandomResizedCrop(224),  # Resize and crop to 224x224
    #         transforms.RandomHorizontalFlip(),  # Randomly flip horizontally
    #         transforms.ToTensor(),  # Convert to a PyTorch tensor
    #         transforms.Normalize(
    #             mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    #         ),  # Normalize using ImageNet statistics
    #     ]
    # )

    train_dataset = DatasetsDataloaders.get_dataset(
        image_paths=train_image_paths,
        labels=train_label_indices,
        transform=None,
        is_train=True,
        N=2,
        M=10,
    )

    train_dataloader = DatasetsDataloaders.get_dataloader(
        train_dataset, batch_size=32, shuffle=True
    )
    print(f"==>> train_dataloader: {len(train_dataloader)}")

    for idx, (images, labels) in enumerate(val_dataloader):
        print(f"==>> images.shape: {images.shape}")
        print(f"==>> labels.shape: {labels}")
        break

    for idx, (images, labels) in enumerate(train_dataloader):
        print(f"==>> images.shape: {images.shape}")
        print(f"==>> labels.shape: {labels}")
        break
