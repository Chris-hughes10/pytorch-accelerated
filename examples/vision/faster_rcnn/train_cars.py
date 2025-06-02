import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from evaluation.calculate_map_callback import CalculateMeanAveragePrecisionCallback
from frcnn.dataset import FasterRCNNDataset, get_eval_transforms, get_train_transforms
from frcnn.model import create_frcnn_model
from frcnn.trainer import FasterRCNNTrainer, faster_rcnn_collate_fn
from func_to_script import script
from PIL import Image
from torch.utils.data import Dataset

from pytorch_accelerated.callbacks import get_default_callbacks
from pytorch_accelerated.schedulers import CosineLrScheduler


def load_cars_df(annotations_file_path, images_path):
    all_images = sorted(set([p.parts[-1] for p in images_path.iterdir()]))
    image_id_to_image = {i: im for i, im in enumerate(all_images)}
    image_to_image_id = {v: k for k, v in image_id_to_image.items()}

    annotations_df = pd.read_csv(annotations_file_path)
    annotations_df.loc[:, "class_name"] = "car"
    annotations_df.loc[:, "has_annotation"] = True

    # add 100 empty images to the dataset
    empty_images = sorted(set(all_images) - set(annotations_df.image.unique()))
    non_annotated_df = pd.DataFrame(list(empty_images)[:100], columns=["image"])
    non_annotated_df.loc[:, "has_annotation"] = False
    non_annotated_df.loc[:, "class_name"] = "background"

    df = pd.concat((annotations_df, non_annotated_df))

    # As 0 is the background class, class ids must start at 1
    class_id_to_label = {
        i + 1: c
        for i, c in enumerate(df.query("has_annotation == True").class_name.unique())
    }
    class_label_to_id = {v: k for k, v in class_id_to_label.items()}

    df["image_id"] = df.image.map(image_to_image_id)
    df["class_id"] = df.class_name.map(class_label_to_id)

    file_names = tuple(df.image.unique())
    random.seed(42)
    validation_files = set(random.sample(file_names, int(len(df) * 0.2)))
    train_df = df[~df.image.isin(validation_files)]
    valid_df = df[df.image.isin(validation_files)]

    lookups = {
        "image_id_to_image": image_id_to_image,
        "image_to_image_id": image_to_image_id,
        "class_id_to_label": class_id_to_label,
        "class_label_to_id": class_label_to_id,
    }
    return train_df, valid_df, lookups


class DatasetAdaptor(Dataset):
    def __init__(
        self,
        images_dir_path,
        annotations_dataframe,
        transforms=None,
    ):
        self.images_dir_path = Path(images_dir_path)
        self.annotations_df = annotations_dataframe
        self.transforms = transforms

        self.image_idx_to_image_id = {
            idx: image_id
            for idx, image_id in enumerate(self.annotations_df.image_id.unique())
        }
        self.image_id_to_image_idx = {
            v: k for k, v in self.image_idx_to_image_id.items()
        }

    def __len__(self) -> int:
        return len(self.image_idx_to_image_id)

    def __getitem__(self, index):
        image_id = self.image_idx_to_image_id[index]
        image_info = self.annotations_df[self.annotations_df.image_id == image_id]
        file_name = image_info.image.values[0]
        assert image_id == image_info.image_id.values[0]

        image = Image.open(self.images_dir_path / file_name).convert("RGB")
        image = np.array(image)

        if image_info.has_annotation.any():
            xyxy_bboxes = image_info[["xmin", "ymin", "xmax", "ymax"]].values
            class_ids = image_info["class_id"].values
        else:
            xyxy_bboxes = np.array([])
            class_ids = np.array([])

        if self.transforms is not None:
            transformed = self.transforms(
                image=image, bboxes=xyxy_bboxes, labels=class_ids
            )
            image = transformed["image"]
            xyxy_bboxes = np.array(transformed["bboxes"])
            class_ids = np.array(transformed["labels"])

        return image, xyxy_bboxes, class_ids, image_id


DATA_PATH = Path("/".join(Path(__file__).absolute().parts[:-3])) / "data/cars"


@script
def main(
    data_path: str = DATA_PATH,
    image_size: int = 800,
    num_epochs: int = 30,
    batch_size: int = 8,
):
    data_path = Path(data_path)
    images_path = data_path / "training_images"
    annotations_file_path = data_path / "annotations.csv"

    train_df, valid_df, lookups = load_cars_df(annotations_file_path, images_path)

    train_ds = DatasetAdaptor(
        images_path,
        train_df,
    )
    eval_ds = DatasetAdaptor(images_path, valid_df)

    train_dataset = FasterRCNNDataset(train_ds, transforms=get_train_transforms())
    eval_dataset = FasterRCNNDataset(eval_ds, transforms=get_eval_transforms())

    num_classes = 1

    model = create_frcnn_model(num_classes=num_classes, image_size=image_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

    trainer = FasterRCNNTrainer(
        model=model,
        optimizer=optimizer,
        callbacks=[
            CalculateMeanAveragePrecisionCallback.create_from_targets_df(
                targets_df=valid_df.query("has_annotation == True")[
                    ["image_id", "xmin", "ymin", "xmax", "ymax", "class_id"]
                ],
                image_ids=set(valid_df.image_id.unique()),
                iou_threshold=0.2,
            ),
            *get_default_callbacks(progress_bar=True),
        ],
    )

    trainer.train(
        num_epochs=num_epochs,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        per_device_batch_size=batch_size,
        create_scheduler_fn=CosineLrScheduler.create_scheduler_fn(
            num_warmup_epochs=5,
            num_cooldown_epochs=5,
        ),
        collate_fn=faster_rcnn_collate_fn,
    )


if __name__ == "__main__":
    main()
