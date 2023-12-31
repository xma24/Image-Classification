import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import timm
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision import transforms

from ImageClassification.datasets_dataloaders import (
    DatasetsDataloaders,
    DatasetsDataloadersUtils,
)


class ModelUtils:
    def get_val_dataloader(val_images_txtfile, val_labels_txtfile, batch_size):
        # val_images_txtfile = os.path.join("./work_dirs/val_images.txt")
        # val_labels_txtfile = os.path.join("./work_dirs/val_labels.txt")
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
            val_dataset, batch_size=batch_size, shuffle=False
        )
        print(f"==>> val_dataloader: {len(val_dataloader)}")

        return val_dataloader, class_to_idx

    def get_train_dataloader(train_images_txtfile, train_labels_txtfile, batch_size):
        # train_images_txtfile = os.path.join("./work_dirs/train_images.txt")
        # train_labels_txtfile = os.path.join("./work_dirs/train_labels.txt")
        train_image_paths = DatasetsDataloadersUtils.read_list_from_txtfile(
            train_images_txtfile
        )
        train_labels = DatasetsDataloadersUtils.read_list_from_txtfile(
            train_labels_txtfile
        )
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
            train_dataset, batch_size=batch_size, shuffle=True
        )
        print(f"==>> train_dataloader: {len(train_dataloader)}")
        return train_dataloader, class_to_idx


class Model(pl.LightningModule):
    def __init__(
        self,
        learning_rate,
        num_classes,
        model_name,
        is_train_pl=True,
    ):
        super(Model, self).__init__()
        self.learning_rate = learning_rate
        self.num_classes = num_classes

        pretrained_models = timm.list_models(pretrained=True)
        print(f"==>> pretrained_models: {pretrained_models}")

        self.model = timm.create_model(model_name, pretrained=True)

        if is_train_pl:
            self.model.classifier = nn.Linear(
                self.model.classifier.in_features, num_classes
            )

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log("val_loss", loss)
