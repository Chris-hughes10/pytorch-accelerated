import json
import sys
from typing import Collection, Dict, List
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

XMIN_COL = "xmin"
YMIN_COL = "ymin"
XMAX_COL = "xmax"
YMAX_COL = "ymax"
CLASS_ID_COL = "class_id"
SCORE_COL = "score"
BOX_WIDTH_COL = "w"
BOX_HEIGHT_COL = "h"
IMAGE_ID_COL = "image_id"


class Silencer:
    def __init__(self):
        self.save_stdout = sys.stdout

    def start(self):
        sys.stdout = MagicMock()

    def stop(self):
        sys.stdout = self.save_stdout

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class COCOMeanAveragePrecision:
    """
    Computes the Mean-Average-Precision (mAP) for object detection predictions and ground truth targets.

    This class provides an interface to the mAP implementation included with pycocotools, which is the standard
    implementation for the benchmarking of object detection models in academic papers.

    """

    # Box area range is a concept needed for benchmarking models, we do not need that.
    # Thus, we pick one that can just fit any prediction.
    AREA_RANGE = np.array([0**2, 1e5**2])
    AREA_RANGE_LABEL = "all"
    # Maximum number of predictions we account for each image.
    MAX_PREDS = 100  # TODO was this the default?

    def __init__(self, iou_threshold: float = None, verbose=False):
        """
        :param iou_threshold: If set, the IoU threshold at which mAP will be calculated. Otherwise, the COCO default range of IoU thresholds will be used.
        :param verbose: If True, display the output provided by pycocotools, containing the average precision and recall across a range of box sizes.
        """
        self.foreground_threshold = iou_threshold
        self.verbose = verbose
        self.silencer = Silencer()

    def compute(self, targets_json: dict, predictions_json: List[dict]):
        """
        Calculate mAP from COCO-formatted dictionaries containing predictions and targets.

        .. Note:: If no predictions are made, -1 will be returned.

        :param targets_json: a dictionary with the keys "images", "categories" and "annotations"
        :param predictions_json: list of dictionaries with annotations in coco format

        """
        if len(predictions_json) == 0:
            # If there are no predictions (sometimes on error impact), return -1.
            return -1

        if not self.verbose:
            self.silencer.start()

        coco_eval = self._build_coco_eval(targets_json, predictions_json)
        coco_eval.evaluate()
        coco_eval.accumulate()
        if self.foreground_threshold is None:
            coco_eval.summarize()
            mAP = coco_eval.stats[0]
        else:
            mAP = self._compute(coco_eval)

        if not self.verbose:
            self.silencer.stop()

        return mAP

    def compute_from_dfs(
        self,
        targets_df: pd.DataFrame,
        predictions_df: pd.DataFrame,
    ) -> float:
        """
        Calculate mAP from dataframes containing predictions and targets.

        .. Note:: If no predictions are made, -1 will be returned.

        :param targets_df: DF w/ cols: ["image_id", "xmin", "ymin", "xmax", "ymax", "class_id"]
        :param predictions_df: DF w/ cols: ["image_id", "xmin", "ymin", "xmax", "ymax", "class_id", "score"]

        """
        if len(predictions_df) == 0:
            # If there are no predictions (sometimes on error impact), return -1.
            return -1

        image_ids = set(targets_df[IMAGE_ID_COL].unique())
        image_ids.update(predictions_df[IMAGE_ID_COL].unique())

        targets, preds = self._format_inputs(targets_df, predictions_df, image_ids)
        # Silence all the garbage prints that cocotools spits out
        # with silencer():
        map = self.compute(targets, preds)
        return map

    def _format_inputs(self, targets_df, preds_df, image_ids):
        preds = self._format_box_df_for_cocotools(preds_df, is_preds=True)
        # Targets are expected to carry extra information
        targets = self.create_targets_coco_json_from_df(targets_df, image_ids)

        return targets, preds

    @classmethod
    def create_targets_coco_json_from_df(
        cls, targets_df: pd.DataFrame, image_ids: Collection
    ) -> dict:
        """
        Create a COCO-formatted annotations dictionary that can be used for evaluation.

        :param targets_df: DF w/ cols: ["image_id", "xmin", "ymin", "xmax", "ymax", "class_id"]
        :param image_ids: A collection of all image ids in the dataset, including those without annotations.

        :return a dictionary with the keys "images", "categories" and "annotations"

        """
        target_anns = cls._format_box_df_for_cocotools(targets_df)
        targets = {
            "images": [{"id": id_} for id_ in set(image_ids)],
            "categories": [{"id": cat} for cat in targets_df[CLASS_ID_COL].unique()],
            "annotations": target_anns,
        }
        return targets

    @classmethod
    def create_predictions_coco_json_from_df(
        cls, predictions_df: pd.DataFrame
    ) -> List[dict]:
        """
        Create a COCO-formatted predictions list of dictionaries that can be used for evaluation.

        :param predictions_df: DF w/ cols: ["image_id", "xmin", "ymin", "xmax", "ymax", "class_id", "score"]
        :param image_ids: A collection of all image ids in the dataset, including those without annotations.

        :return a list of dictionaries with annotations in coco format

        """
        return cls._format_box_df_for_cocotools(predictions_df, is_preds=True)

    @staticmethod
    def _format_box_df_for_cocotools(
        box_df: pd.DataFrame, is_preds: bool = False
    ) -> List[Dict]:
        # `box_df` is either a `targets_df` or a `preds_df`
        box_df = box_df.copy()  # Ensure no side effects
        box_df[BOX_WIDTH_COL] = box_df[XMAX_COL] - box_df[XMIN_COL]
        box_df[BOX_HEIGHT_COL] = box_df[YMAX_COL] - box_df[YMIN_COL]
        box_df = box_df.sort_values(
            [IMAGE_ID_COL, CLASS_ID_COL], ascending=[True, True]
        )

        ann_records = json.loads(box_df.to_json(orient="records"))

        formatted = [
            {
                "id": i,
                "image_id": r[IMAGE_ID_COL],
                "category_id": int(r[CLASS_ID_COL]),
                "bbox": [r[XMIN_COL], r[YMIN_COL], r[BOX_WIDTH_COL], r[BOX_HEIGHT_COL]],
                "iscrowd": False,
                "area": r[BOX_WIDTH_COL] * r[BOX_HEIGHT_COL],
            }
            for i, r in enumerate(ann_records, start=1)
        ]
        if is_preds:
            # preds need a "score" field
            for r, a in zip(ann_records, formatted):
                a["score"] = r[SCORE_COL]
                a.pop("id")
        return formatted

    def _build_coco_eval(self, targets, preds):
        """Build the COCOeval object we need to leverage pycocotools computation"""
        coco_targets = COCO()
        coco_targets.dataset = targets
        coco_targets.createIndex()
        coco_preds = coco_targets.loadRes(preds)
        coco_eval = COCOeval(cocoGt=coco_targets, cocoDt=coco_preds, iouType="bbox")

        if self.foreground_threshold is not None:
            coco_eval.params.iouThrs = np.array(
                [self.foreground_threshold]
            )  # Single IoU threshold
            coco_eval.params.areaRng = np.array([self.AREA_RANGE])
            coco_eval.params.areaRngLbl = [self.AREA_RANGE_LABEL]
            # Single maximum number of predictions to account for
            coco_eval.params.maxDets = np.array([self.MAX_PREDS])
        return coco_eval

    def _compute(self, coco_eval):
        """
        Perform the mAP computation for custom IoU thresholds; extracted from non-flexible `COCOeval.summarize` method.

        """
        p = coco_eval.params

        aind = [
            i for i, aRng in enumerate(p.areaRngLbl) if aRng == self.AREA_RANGE_LABEL
        ]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == self.MAX_PREDS]
        # dimension of precision: [TxRxKxAxM]
        s = coco_eval.eval["precision"]
        # IoU
        t = np.where(self.foreground_threshold == p.iouThrs)[0]
        s = s[t]
        s = s[:, :, :, aind, mind]
        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])
        return mean_s
