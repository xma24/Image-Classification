import json
import multiprocessing
import os
import random
import sys
import numpy as np
import shutil

from ImageClassification.rand_augment import RandAugment
from PIL import Image
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
    def save_listOflist_to_txt(list_of_lists, file_path):
        with open(file_path, "w") as file:
            for sublist in list_of_lists:
                # Convert each element in the sublist to a string and join with commas
                line = ",".join(map(str, sublist))

                # Write the line to the file
                file.write(line + "\n")

    @staticmethod
    def read_txt_to_listOflist(file_path):
        result = []
        with open(file_path, "r") as file:
            for line in file:
                # Split each line by commas and convert elements to integers
                sublist = [int(x) for x in line.strip().split(",")]

                # Append the sublist to the result list
                result.append(sublist)
        return result

    @staticmethod
    def voc_train_single_class_summary(voc_cls_train_labels_txt_path):
        train_cls_classid_list = DatasetsDataloadersUtils.read_txt_to_listOflist(
            voc_cls_train_labels_txt_path
        )
        flat_list = [item for sublist in train_cls_classid_list for item in sublist]
        unique_numbers = set(flat_list)
        unique_numbers_list = list(unique_numbers)
        print(f"==>> unique_numbers_list: {unique_numbers_list}")

        train_single_class_summary_dict = {}
        for train_label_list_per in train_cls_classid_list:
            if len(train_label_list_per) == 1:
                train_label_per = train_label_list_per[0]
                if train_label_per in train_single_class_summary_dict:
                    train_single_class_summary_dict[train_label_per] += 1
                else:
                    train_single_class_summary_dict[train_label_per] = 1
        train_single_class_summary_dict = {
            key: train_single_class_summary_dict[key]
            for key in sorted(train_single_class_summary_dict)
        }
        print(
            f"==>> train_single_class_summary_dict: {train_single_class_summary_dict}"
        )

    @staticmethod
    def voc_get_single_class_train_image_path(
        voc_cls_train_images_txt_path,
        voc_cls_train_labels_txt_path,
        voc_single_cls_train_images_txt_path,
        voc_single_cls_train_labels_txt_path,
    ):
        train_cls_classid_list = DatasetsDataloadersUtils.read_txt_to_listOflist(
            voc_cls_train_labels_txt_path
        )
        train_cls_image_path_list = DatasetsDataloadersUtils.read_list_from_txtfile(
            voc_cls_train_images_txt_path
        )

        single_train_image_path_list = []
        single_train_label_list = []
        for idx, train_label_list_per in enumerate(train_cls_classid_list):
            if len(train_label_list_per) == 1:
                train_label_per = train_label_list_per[0]
                single_train_label_list.append(train_label_per)
                single_train_image_path_list.append(train_cls_image_path_list[idx])
        DatasetsDataloadersUtils.save_list_to_txtfile(
            single_train_image_path_list, voc_single_cls_train_images_txt_path
        )
        DatasetsDataloadersUtils.save_list_to_txtfile(
            single_train_label_list, voc_single_cls_train_labels_txt_path
        )

    @staticmethod
    def voc_get_single_class_val_image_path(
        voc_cls_val_images_txt_path,
        voc_cls_val_labels_txt_path,
        voc_single_cls_val_images_txt_path,
        voc_single_cls_val_labels_txt_path,
    ):
        val_cls_classid_list = DatasetsDataloadersUtils.read_txt_to_listOflist(
            voc_cls_val_labels_txt_path
        )
        val_cls_image_path_list = DatasetsDataloadersUtils.read_list_from_txtfile(
            voc_cls_val_images_txt_path
        )

        single_val_image_path_list = []
        single_val_label_list = []
        for idx, val_label_list_per in enumerate(val_cls_classid_list):
            if len(val_label_list_per) == 1:
                val_label_per = val_label_list_per[0]
                single_val_label_list.append(val_label_per)
                single_val_image_path_list.append(val_cls_image_path_list[idx])
        DatasetsDataloadersUtils.save_list_to_txtfile(
            single_val_image_path_list, voc_single_cls_val_images_txt_path
        )
        DatasetsDataloadersUtils.save_list_to_txtfile(
            single_val_label_list, voc_single_cls_val_labels_txt_path
        )

    @staticmethod
    def voc_color_map(N=256, normalized=False):
        voc_labelname_to_classid_dict = {
            "background": 0,  # 0
            "aeroplane": 1,  # 1
            "bicycle": 2,  # 2
            "bird": 3,  # 3
            "boat": 4,  # 4
            "bottle": 5,  # 5
            "bus": 6,  # 6
            "car": 7,  # 7
            "cat": 8,  # 8
            "chair": 9,  # 9
            "cow": 10,  # 10
            "diningtable": 11,  # 11
            "dog": 12,  # 12
            "horse": 13,  # 13
            "motorbike": 14,  # 14
            "person": 15,  # 15
            "pottedplant": 16,  # 16
            "sheep": 17,  # 17
            "sofa": 18,  # 18
            "train": 19,  # 19
            "tv/monitor": 20,  # 20
            "void/unlabelled": 255,  # 255
        }

        def bitget(byteval, idx):
            return (byteval & (1 << idx)) != 0

        dtype = "float32" if normalized else "uint8"
        cmap = np.zeros((N, 3), dtype=dtype)
        for i in range(N):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7 - j)
                g = g | (bitget(c, 1) << 7 - j)
                b = b | (bitget(c, 2) << 7 - j)
                c = c >> 3

            cmap[i] = np.array([r, g, b])

        cmap = cmap / 255 if normalized else cmap
        print(f"==>> cmap: {cmap}")

        voc_classid_to_labelname_dict = {}
        for key, value in voc_labelname_to_classid_dict.items():
            voc_classid_to_labelname_dict[value] = key

        voc_classid_to_color_dict = {}
        for class_id in list(voc_classid_to_labelname_dict.keys()):
            per_color = cmap[class_id]
            per_color_string = f"{per_color[0]}, {per_color[1]}, {per_color[2]}"
            voc_classid_to_color_dict[class_id] = per_color_string

        voc_color_to_classid_dict = {}
        for key, value in voc_classid_to_color_dict.items():
            voc_color_to_classid_dict[value] = key

        return (
            voc_labelname_to_classid_dict,
            voc_classid_to_labelname_dict,
            voc_classid_to_color_dict,
            voc_color_to_classid_dict,
        )

    @staticmethod
    def voc_get_unique_classid(image_path, voc_color_to_classid_dict):
        img = Image.open(image_path)
        img = img.convert("RGB")
        pixels = list(img.getdata())
        unique_colors = set()
        for pixel in pixels:
            unique_colors.add(pixel)
        unique_colors_list = list(unique_colors)

        unique_classid_list = []
        for color in unique_colors_list:
            color_string_per = f"{color[0]}, {color[1]}, {color[2]}"
            # print(f"==>> color_string_per: {color_string_per}")
            if color_string_per in voc_color_to_classid_dict:
                classid_per = voc_color_to_classid_dict[color_string_per]
                # print(f"==>> classid_per: {classid_per}")
                if classid_per != 0 and classid_per != 255:
                    unique_classid_list.append(classid_per - 1)
            # else:
            # print(f"Not find the color in voc cmap.")
            # print(f"==>> color_string_per: {color_string_per}")
        # print(f"==>> unique_classid_list: {unique_classid_list}")

        return unique_classid_list

    @staticmethod
    def voc_clean_mask(voc_mask_folder, clean_mask_folder):
        os.makedirs(clean_mask_folder, exist_ok=True)

        (
            voc_labelname_to_classid_dict,
            voc_classid_to_labelname_dict,
            voc_classid_to_color_dict,
            voc_color_to_classid_dict,
        ) = DatasetsDataloadersUtils.voc_color_map()
        print(f"==>> voc_labelname_to_classid_dict: {voc_labelname_to_classid_dict}")
        print(f"==>> voc_classid_to_labelname_dict: {voc_classid_to_labelname_dict}")
        print(f"==>> voc_classid_to_color_dict: {voc_classid_to_color_dict}")
        print(f"==>> voc_color_to_classid_dict: {voc_color_to_classid_dict}")

        mask_image_path_list = DatasetsDataloadersUtils.get_images_in_folder(
            voc_mask_folder
        )
        mask_image_path_list = sorted(mask_image_path_list)

        for idx, mask_image_path in enumerate(mask_image_path_list):
            if idx % 100 == 0:
                print(f"==>> idx: {idx}")

            clean_mask_image_path = os.path.join(
                clean_mask_folder, mask_image_path.split("/")[-1]
            )

            mask_image = Image.open(mask_image_path)
            #

            if mask_image.mode == "L":
                print(f"==>> mask_image_path: {mask_image_path}")
                rgb_image = Image.new("RGB", mask_image.size)
                for class_id, color_str in voc_classid_to_color_dict.items():
                    color_str_list = color_str.split(",")
                    color = (
                        int(color_str_list[0]),
                        int(color_str_list[1]),
                        int(color_str_list[2]),
                    )
                    color_image = Image.new("RGB", mask_image.size, color)
                    rgb_image.paste(
                        color_image,
                        mask=mask_image.convert("L").point(
                            lambda p: p == class_id and 255
                        ),
                    )
                rgb_image.save(clean_mask_image_path)
            else:
                shutil.copyfile(mask_image_path, clean_mask_image_path)

    @staticmethod
    def voc_get_cls_image_labels_from_seg_masks(
        voc_image_folder,
        voc_mask_folder,
        voc_seg_train_images_txt_path,
        voc_seg_val_images_txt_path,
        voc_cls_train_images_txt_path,
        voc_cls_train_labels_txt_path,
        voc_cls_val_images_txt_path,
        voc_cls_val_labels_txt_path,
    ):
        (
            voc_labelname_to_classid_dict,
            voc_classid_to_labelname_dict,
            voc_classid_to_color_dict,
            voc_color_to_classid_dict,
        ) = DatasetsDataloadersUtils.voc_color_map()
        print(f"==>> voc_labelname_to_classid_dict: {voc_labelname_to_classid_dict}")
        print(f"==>> voc_classid_to_labelname_dict: {voc_classid_to_labelname_dict}")
        print(f"==>> voc_classid_to_color_dict: {voc_classid_to_color_dict}")
        print(f"==>> voc_color_to_classid_dict: {voc_color_to_classid_dict}")

        mask_image_path_list = DatasetsDataloadersUtils.get_images_in_folder(
            voc_mask_folder
        )
        mask_image_path_list = sorted(mask_image_path_list)

        train_labelname_list = DatasetsDataloadersUtils.read_list_from_txtfile(
            voc_seg_train_images_txt_path
        )
        val_labelname_list = DatasetsDataloadersUtils.read_list_from_txtfile(
            voc_seg_val_images_txt_path
        )

        train_cls_classid_list = []
        train_cls_image_path_list = []
        val_cls_classid_list = []
        val_cls_image_path_list = []

        for idx, mask_image_path in enumerate(mask_image_path_list):
            if idx % 100 == 0:
                print(f"==>> idx: {idx}")

            mask_image_name = mask_image_path.split("/")[-1].split(".")[0]
            image_path = os.path.join(voc_image_folder, mask_image_name + ".jpg")

            unique_classid_per_image = DatasetsDataloadersUtils.voc_get_unique_classid(
                mask_image_path, voc_color_to_classid_dict
            )

            if mask_image_name in train_labelname_list:
                train_cls_image_path_list.append(image_path)
                train_cls_classid_list.append(unique_classid_per_image)
            elif mask_image_name in val_labelname_list:
                val_cls_image_path_list.append(image_path)
                val_cls_classid_list.append(unique_classid_per_image)
            else:
                print(f"Image not in Train or Val.")
                sys.exit()

        flat_list = [item for sublist in train_cls_classid_list for item in sublist]
        unique_numbers = set(flat_list)
        unique_numbers_list = list(unique_numbers)
        print(f"==>> unique_numbers_list: {unique_numbers_list}")

        DatasetsDataloadersUtils.save_list_to_txtfile(
            train_cls_image_path_list, voc_cls_train_images_txt_path
        )
        DatasetsDataloadersUtils.save_listOflist_to_txt(
            train_cls_classid_list, voc_cls_train_labels_txt_path
        )
        DatasetsDataloadersUtils.save_list_to_txtfile(
            val_cls_image_path_list, voc_cls_val_images_txt_path
        )
        DatasetsDataloadersUtils.save_listOflist_to_txt(
            val_cls_classid_list, voc_cls_val_labels_txt_path
        )

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
