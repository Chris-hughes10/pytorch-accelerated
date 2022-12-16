import albumentations as A
import numpy as np
import torch
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset


class FasterRCNNDataset(Dataset):
    """Torch Datset that formats inputs ready for PyTorch Faster-RCNN

    :param base_dataset: Initialized torch dataset that returns a numpy array, np.array bboxes, np.array labels
    """

    def __init__(self, base_dataset: Dataset, transforms=None):
        self.ds = base_dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        image, bboxes, labels, image_idx = self.ds[index]

        if self.transforms:
            # Albumentations expect the images to be numpy arrays
            image = np.array(image)
            transformed = self.transforms(image=image, bboxes=bboxes, labels=labels)
            image = transformed["image"]
            bboxes = np.array(transformed["bboxes"])
            labels = transformed["labels"]

        # ensure image in range [0, 1]
        image = image / 255

        targets = self._format_targets(bboxes, labels)
        return image, targets, torch.tensor(image_idx)

    @staticmethod
    def _format_targets(bboxes: np.array, labels: np.array):
        "Format targets the way Faster-RCNN expects them"
        if bboxes.size == 0:
            # PyTorch Faster-RCNN expects targets to be tensors that fulfill
            #   len(boxes.shape) == 2 & boxes.shape[-1] == 4
            bboxes = torch.empty(0, 4)

        targets = {
            "boxes": torch.as_tensor(bboxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
        }
        return targets


def get_train_transforms():
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Affine(
                scale=(0.8, 1.2),
                translate_percent={
                    "x": (-0.3, 0.3),
                    "y": (-0.3, 0.3),
                },
                shear=(-20, 20),
                p=0.8,
            ),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc", min_area=0, min_visibility=0, label_fields=["labels"]
        ),
    )


def get_eval_transforms():
    return A.Compose(
        [
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc", min_area=0, min_visibility=0, label_fields=["labels"]
        ),
    )
