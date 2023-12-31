import json
import os
import sys

import dill
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from ImageClassification.model import Model, ModelUtils
from PIL import Image


class ImageClassification:
    def __init__(
        self,
        random_state=None,
        model_name="efficientnet_b0",
        lr=1e-3,
        num_gpus=1,
        precision=16,
        log_every_n_steps=10,
        min_epochs=1,
        max_epochs=10,
        strategy="ddp",
        accelerator="gpu",
        train_dataloder=None,
        val_dataloader=None,
        class_to_idx=None,
        val_transform=None,
    ):
        self.random_state = random_state
        self.model_name = model_name
        self.lr = lr
        if num_gpus is None:
            self.num_gpus = torch.cuda.device_count()
        else:
            self.num_gpus = num_gpus
        self.precision = precision
        self.log_every_n_steps = log_every_n_steps
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self.strategy = strategy
        self.accelerator = accelerator
        self.train_dataloder = train_dataloder
        self.val_dataloader = val_dataloader
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        if val_transform is None:
            self.val_transform = transforms.Compose(
                [
                    transforms.Resize(256),  # Resize to 256x256
                    transforms.CenterCrop(224),  # Crop the center to 224x224
                    transforms.ToTensor(),  # Convert to tensor
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],  # Mean values for normalization
                        std=[
                            0.229,
                            0.224,
                            0.225,
                        ],  # Standard deviation values for normalization
                    ),
                ]
            )
        else:
            self.val_transform = val_transform

    def fit(self):
        trainer = pl.Trainer(
            devices=self.num_gpus,
            accelerator=self.accelerator,
            strategy=self.strategy,
            max_epochs=self.max_epochs,
            min_epochs=self.min_epochs,
            log_every_n_steps=self.log_every_n_steps,
            precision=self.precision,
        )

        if self.train_dataloder is None:
            self.model = Model(
                self.lr,
                num_classes=len(self.class_to_idx),
                model_name=self.model_name,
                is_train_pl=False,
            )
            print(f"Not train the model. Use pretrained model.")

            with open("imagenet_class_index.json") as f:
                idx_to_class_strkey = json.load(f)
                self.idx_to_class = {}
                for key, value in idx_to_class_strkey.items():
                    self.idx_to_class[int(key)] = value[0] + "-" + value[1]
                # print(f"==>> self.idx_to_class: {self.idx_to_class}")

        else:
            if len(self.class_to_idx) > 0 and self.class_to_idx is not None:
                self.model = Model(
                    self.lr,
                    num_classes=len(self.class_to_idx),
                    model_name=self.model_name,
                    is_train_pl=True,
                )
            else:
                print(f"Model config error.")
                sys.exit()

            trainer.fit(
                self.model,
                train_dataloaders=self.train_dataloder,
                val_dataloaders=self.val_dataloader,
            )

    @torch.no_grad()
    def transform(self, input_data):
        self.model.eval()
        if isinstance(input_data, str):
            input_data = [input_data]

        transformed_images_list = []
        for image_path in input_data:
            image = Image.open(image_path)
            transformed_image = self.val_transform(image)
            transformed_images_list.append(transformed_image)
        transformed_image = torch.stack(transformed_images_list, dim=0)

        outputs = self.model(transformed_image)

        if len(outputs.shape) == 0:  # Handle the case of a 0-dimensional tensor
            outputs = outputs.unsqueeze(0)  # Convert to a 1-dimensional tensor
        if len(outputs.shape) == 1:
            outputs = outputs.unsqueeze(0)  # Convert to a 2-dimensional tensor

        # Use the outputs to get the top-1 and top-5 predictions
        _, top1_indices = torch.topk(outputs, k=1, dim=1)
        _, top5_indices = torch.topk(outputs, k=5, dim=1)

        # Get the probabilities of top-1 and top-5 predictions
        top1_probs = torch.softmax(outputs, dim=1).gather(1, top1_indices)
        top5_probs = torch.softmax(outputs, dim=1).gather(1, top5_indices)

        # Convert tensors to Python lists
        top1_indices = top1_indices.tolist()
        top5_indices = top5_indices.tolist()
        top1_probs = top1_probs.tolist()
        top5_probs = top5_probs.tolist()

        # Map indices to class names
        prediction_top1 = [self.idx_to_class[idx[0]] for idx in top1_indices]
        prediction_top5 = [
            [self.idx_to_class[idx] for idx in indices] for indices in top5_indices
        ]

        # Combine predictions, indices, and probabilities
        predictions_with_info = []
        for i in range(len(prediction_top1)):
            prediction_info = {
                "image_path": input_data[i],
                "class_name": prediction_top1[i],
                "class_index": top1_indices[i],
                "probability": top1_probs[i],
                "top_5": prediction_top5[i],
            }
            predictions_with_info.append(prediction_info)

        return predictions_with_info

    def save(self, path):
        with open(path, "wb") as oup:
            dill.dump(self, oup)
        print(f"Pickled model project at {path}")


def load_model(path):
    print("Loading model object from pickled file.")
    with open(path, "rb") as inp:
        return dill.load(inp)
