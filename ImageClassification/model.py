import logging
import os
import sys
import time

import pytorch_lightning as pl
import timm
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from ImageClassification.datasets_dataloaders import (
    DatasetsDataloaders,
    DatasetsDataloadersUtils,
)
from torch.optim.lr_scheduler import CosineAnnealingLR

logger_initialized = {}


class ModelUtils:
    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def get_classification_metrics(num_classes):
        train_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes, average="none"
        )

        train_precision = torchmetrics.Precision(
            task="multiclass", num_classes=num_classes, average="none"
        )

        train_recall = torchmetrics.Recall(
            task="multiclass", num_classes=num_classes, average="none"
        )

        val_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes, average="none"
        )

        val_precision = torchmetrics.Precision(
            task="multiclass", num_classes=num_classes, average="none"
        )

        val_recall = torchmetrics.Recall(
            task="multiclass", num_classes=num_classes, average="none"
        )

        test_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes, average="none"
        )

        test_precision = torchmetrics.Precision(
            task="multiclass", num_classes=num_classes, average="none"
        )

        test_recall = torchmetrics.Recall(
            task="multiclass", num_classes=num_classes, average="none"
        )

        return (
            train_accuracy,
            train_precision,
            train_recall,
            val_accuracy,
            val_precision,
            val_recall,
            test_accuracy,
            test_precision,
            test_recall,
        )

    @staticmethod
    def get_logger(name, log_file=None, log_level=logging.INFO, file_mode="w"):
        logger = logging.getLogger(name)
        if name in logger_initialized:
            return logger
        for logger_name in logger_initialized:
            if name.startswith(logger_name):
                return logger
        for handler in logger.root.handlers:
            if type(handler) is logging.StreamHandler:
                handler.setLevel(logging.ERROR)

        stream_handler = logging.StreamHandler()
        handlers = [stream_handler]

        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0

        # only rank 0 will add a FileHandler
        if rank == 0 and log_file is not None:
            file_handler = logging.FileHandler(log_file, file_mode)
            handlers.append(file_handler)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        for handler in handlers:
            handler.setFormatter(formatter)
            handler.setLevel(log_level)
            logger.addHandler(handler)

        if rank == 0:
            logger.setLevel(log_level)
        else:
            logger.setLevel(logging.ERROR)

        logger_initialized[name] = True

        return logger

    @staticmethod
    def get_root_logger(log_file=None, log_level=logging.INFO):
        logger = ModelUtils.get_logger(
            name="ImgCls", log_file=log_file, log_level=log_level
        )

        return logger

    @staticmethod
    def console_logger_start():
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())

        os.makedirs("./work_dirs/", exist_ok=True)
        console_log_file = os.path.join("./work_dirs/", f"{timestamp}.log")
        console_logger = ModelUtils.get_root_logger(
            log_file=console_log_file, log_level=logging.INFO
        )
        return console_logger


class Model(pl.LightningModule):
    def __init__(
        self,
        batch_size,
        learning_rate,
        num_classes,
        model_name,
        is_train_pl=True,
        optimizer="Adam",
        scheduler="cosAnn",
        single_lr=True,
        backbone_lr=None,
        max_epochs=None,
    ):
        super(Model, self).__init__()
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.single_lr = single_lr
        if single_lr:
            self.backbone_lr = learning_rate
        else:
            if backbone_lr is None:
                print(f"Backbone lr is with error.")
                sys.exit()
            self.backbone_lr = backbone_lr
        self.module_lr_dict = dict(backbone=self.backbone_lr)
        self.max_epochs = max_epochs

        self.log_config_step = dict(
            on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True
        )
        self.log_config_epoch = dict(
            on_epoch=True, prog_bar=True, logger=True, sync_dist=True
        )

        pretrained_models = timm.list_models(pretrained=True)
        print(f"==>> pretrained_models: {pretrained_models}")

        self.model = timm.create_model(model_name, pretrained=True)

        if is_train_pl:
            self.model.classifier = nn.Linear(
                self.model.classifier.in_features, num_classes
            )
        if num_classes is not None:
            (
                self.train_accuracy,
                self.train_precision,
                self.train_recall,
                self.val_accuracy,
                self.val_precision,
                self.val_recall,
                self.test_accuracy,
                self.test_precision,
                self.test_recall,
            ) = ModelUtils.get_classification_metrics(num_classes)
        else:
            print(f"Classes not provided.")
        self.console_logger = ModelUtils.console_logger_start()

    def lr_logging(self):
        """+++ Capture the learning rates and log it out using logger;"""
        lightning_optimizer = self.optimizers()
        param_groups = lightning_optimizer.optimizer.param_groups

        for param_group_idx in range(len(param_groups)):
            sub_param_group = param_groups[param_group_idx]

            """>>>
            print("==>> sub_param_group: ", sub_param_group.keys())
            # ==>> sub_param_group:  dict_keys(['params', 'lr', 'momentum', 'dampening', 'weight_decay', 'nesterov', 'maximize', 'initial_lr'])
            """

            sub_lr_name = "lr/lr_" + str(param_group_idx)
            """>>>
            print("lr: {}, {}".format(sub_lr_name, sub_param_group["lr"]))
            # lr: lr_0, 0.001
            # lr: lr_1, 0.08
            """

            self.log(
                sub_lr_name,
                sub_param_group["lr"],
                batch_size=self.batch_size,
                **self.log_config_step,
            )

    def different_lr(self, module_lr_dict, lr):
        def is_key_included(module_name, n):
            return module_name in n

        def is_any_key_match(module_lr_dict, n):
            indicator = False
            for key in module_lr_dict.keys():
                if key in n:
                    indicator = True
            return indicator

        params = list(self.named_parameters())

        grouped_parameters = [
            {
                "params": [
                    p for n, p in params if not is_any_key_match(module_lr_dict, n)
                ],
                "lr": lr,
            }
        ]

        for key in module_lr_dict.keys():
            sub_param_list = []
            for n, p in params:
                if is_key_included(key, n):
                    if module_lr_dict[key] == 0.0:
                        p.requires_grad = False
                    sub_param_list.append(p)
            sub_parameters = {"params": sub_param_list, "lr": module_lr_dict[key]}
            grouped_parameters.append(sub_parameters)

        return grouped_parameters

    def get_groupparamters(self):
        if self.single_lr:
            print(f"Using a single learning rate for all parameters")
            grouped_parameters = [{"params": self.parameters()}]
        else:
            print("Using different learning rates for all parameters")
            grouped_parameters = self.different_lr(
                self.module_lr_dict, self.learning_rate
            )
        return grouped_parameters

    def get_optim(self, grouped_parameters):
        if self.optimizer == "Adam":
            lr = 0.001
            betas = (0.9, 0.999)
            eps = 1e-08
            weight_decay = 0.0

            optimizer = torch.optim.Adam(
                grouped_parameters,
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
            )
        elif self.optimizer == "SGD":
            momentum = 0.9
            weight_decay = 0.0001

            optimizer = torch.optim.SGD(
                grouped_parameters,
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
            )
        optimizer.zero_grad()
        return optimizer

    def configure_optimizers(self):
        grouped_parameters = self.get_groupparamters()
        optimizer = self.get_optim(grouped_parameters)
        if self.scheduler == "cosAnn":
            eta_min = 0.0

            T_max = self.max_epochs
            last_epoch = -1

            sch = CosineAnnealingLR(
                optimizer, T_max=T_max, eta_min=eta_min, last_epoch=last_epoch
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": sch, "monitor": "train_loss"},
            }
        else:
            return optimizer

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        self.lr_logging()

        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log(
            "train_loss",
            loss,
            batch_size=self.batch_size,
            **self.log_config_step,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log("val_loss", loss, sync_dist=True)

        # Convert model output (logits) to probability distribution using softmax
        # and then get the index of the maximum value in each prediction (the predicted class)
        model_predictions = torch.argmax(nn.functional.softmax(y_hat, dim=1), dim=1)

        # Ground truth labels are simply 'y'
        gt_labels = y

        # Update metrics
        self.val_accuracy.update(model_predictions, gt_labels)
        self.val_precision.update(model_predictions, gt_labels)
        self.val_recall.update(model_predictions, gt_labels)

    def on_validation_epoch_end(self):
        """>>> The compute() function will return a list;"""
        val_epoch_accuracy = self.val_accuracy.compute()
        val_epoch_precision = self.val_precision.compute()
        val_epoch_recall = self.val_recall.compute()

        val_epoch_accuracy_mean = torch.mean(val_epoch_accuracy).item()
        val_epoch_precision_mean = torch.mean(val_epoch_precision).item()
        val_epoch_recall_mean = torch.mean(val_epoch_recall).item()

        self.log(
            "val_epoch_accuracy",
            val_epoch_accuracy_mean,
            batch_size=self.batch_size,
            **self.log_config_epoch,
        )

        user_metric = val_epoch_accuracy_mean
        self.log(
            "user_metric",
            user_metric,
            batch_size=self.batch_size,
            **self.log_config_epoch,
        )

        """>>> Plot the results to console using only one gpu since the results on all gpus are the same; """
        if self.global_rank == 0:
            self.console_logger.info("epoch: {0:04d} ".format(self.current_epoch))

            num_display_classes = min(20, self.num_classes)
            for i in range(num_display_classes):
                self.console_logger.info(
                    "{0: <15}, acc: {1:.4f} | precision: {2:.4f} | recall: {3:.4f}".format(
                        str(i),
                        val_epoch_accuracy[i].item(),
                        val_epoch_precision[i].item(),
                        val_epoch_recall[i].item(),
                    )
                )
            self.console_logger.info(
                "acc_mean: {0:.4f} ".format(val_epoch_accuracy_mean)
            )

            self.console_logger.info(
                "precision_mean: {0:.4f} ".format(val_epoch_precision_mean)
            )
            self.console_logger.info(
                "recall_mean: {0:.4f} ".format(val_epoch_recall_mean)
            )

        self.val_accuracy.reset()
        self.val_precision.reset()
        self.val_recall.reset()
