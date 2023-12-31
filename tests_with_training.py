import os

import torch
from ImageClassification import ImageClassification, ModelUtils, load_model

if __name__ == "__main__":
    batch_size = 64

    train_images_txtfile = os.path.join("./work_dirs/train_images.txt")
    train_labels_txtfile = os.path.join("./work_dirs/train_labels.txt")
    train_dataloder, class_to_idx = ModelUtils.get_train_dataloader(
        train_images_txtfile, train_labels_txtfile, batch_size=batch_size
    )
    val_images_txtfile = os.path.join("./work_dirs/val_images.txt")
    val_labels_txtfile = os.path.join("./work_dirs/val_labels.txt")
    val_dataloader, _ = ModelUtils.get_val_dataloader(
        val_images_txtfile, val_labels_txtfile, batch_size=batch_size
    )

    image_classifier = ImageClassification(
        batch_size=batch_size,
        random_state=None,
        model_name="efficientnet_b0",
        lr=1e-3,
        num_gpus=2,
        precision=16,
        log_every_n_steps=10,
        min_epochs=1,
        max_epochs=2,
        strategy="ddp",
        accelerator="gpu",
        train_dataloder=train_dataloder,
        val_dataloader=val_dataloader,
        class_to_idx=class_to_idx,
        val_transform=None,
        optimizer="Adam",
        scheduler="cosAnn",
        single_lr=False,
        backbone_lr=1e-4,
        onnx_file_path="./work_dirs/model.onnx",
        input_sample=torch.randn(1, 3, 224, 224),
    )

    image_classifier.fit()

    os.makedirs("./work_dirs", exist_ok=True)
    project_path = "./work_dirs/image_classifier.pkl"
    image_classifier.save(project_path)

    image_classifier = load_model(project_path)
    predictions = image_classifier.transform(
        [
            "./tests/test_image_n01491361_shark.jpg",
        ]
    )
    print(f"\n******* Results *******")
    for prediction in predictions:
        print(f"==>> prediction: {prediction}\n")
