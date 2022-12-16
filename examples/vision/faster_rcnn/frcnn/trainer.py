from collections import defaultdict

import torch

from pytorch_accelerated import Trainer
from pytorch_accelerated.tracking import LossTracker

FRCNN_PADDING_VALUE = -1


class FasterRCNNTrainer(Trainer):
    def __init__(
        self,
        classifier_weight: float = 1.0,
        box_reg_weight: float = 1.0,
        objectness_weight: float = 1.0,
        rpn_box_reg_weight: float = 1.0,
        **kwargs,
    ):
        # Faster-RCNN has the loss built-in so we need to tell that to pytorch accelerated
        super().__init__(**kwargs, loss_func=None)
        self.loss_weights = {
            "loss_classifier": classifier_weight,
            "loss_box_reg": box_reg_weight,
            "loss_objectness": objectness_weight,
            "loss_rpn_box_reg": rpn_box_reg_weight,
        }
        self._additional_losses = list(self.loss_weights.keys())
        self._additional_loss_trackers = defaultdict(LossTracker)
        self.eval_predictions = []
        self.preds_df = None
        self._metric_prefix = ""

    def train_epoch_start(self):
        super().train_epoch_start()
        self.eval_predictions = []
        self.preds_df = None
        self._metric_prefix = "train_epoch_"
        self._reset_additional_losses()

    def _reset_additional_losses(self):
        for loss_name in self._additional_losses:
            self._additional_loss_trackers[loss_name].reset()

    def evaluation_run_start(self):
        self.eval_predictions = []
        self.preds_df = None
        self._metric_prefix = "evaluation_epoch_"

    def calculate_train_batch_loss(self, batch):
        images, targets, _ = batch
        batch_size = images.size(0)
        losses = self.model(images, targets)

        loss = self._calculate_single_loss(losses)
        losses["loss"] = loss
        self._update_additional_losses(losses, batch_size)

        return {
            "loss": loss,
            "model_outputs": None,
            "batch_size": batch_size,
        }

    def _calculate_single_loss(self, losses):
        """Faster-RCNN outputs multiple losses but we need a single value to optimize

        We use a weighted sum. It is differentiable, simple and it allows to tune importances if
        business case ask for it.

        :param losses: Losses as directly outputed by the model
        :returns loss: Single calculated loss form the input losses
        """
        loss = sum(
            loss_weight * losses[loss_name]
            for loss_name, loss_weight in self.loss_weights.items()
        )
        return loss

    def _update_additional_losses(self, losses, batch_size):
        """Update metrics in the run history for all losses

        :param losses: Dict of losses as outputed by the model with extra "loss" key for single loss
        """

        for loss_name in self._additional_losses:
            self._additional_loss_trackers[loss_name].update(
                self.gather(losses[loss_name]).detach().mean().item(),
                batch_size,
            )

    def train_epoch_end(self):
        self._add_additional_losses_to_run_history()

    def eval_epoch_start(self):
        super().eval_epoch_start()
        self._reset_additional_losses()
        if self._metric_prefix != "evaluation_epoch_":
            self._metric_prefix = "eval_epoch_"

    def calculate_eval_batch_loss(self, batch):
        with torch.no_grad():
            images, targets, image_ids = batch
            batch_size = images.size(0)

            # Losses are only outputed when model is in train mode
            self.model.train()
            losses = self.model(images, targets)
            loss = self._calculate_single_loss(losses)
            losses["loss"] = loss
            self._update_additional_losses(losses, batch_size)

            # Predictions are only outputted when model is in eval mode
            self.model.eval()
            predictions = self.model(images)

            formatted_preds = self._format_predictions(predictions, image_ids)

            gathered_predictions = (
                self.gather(formatted_preds, padding_value=FRCNN_PADDING_VALUE)
                .detach()
                .cpu()
            )

            self.eval_predictions.append(formatted_preds)

            return {
                "loss": loss,
                "model_outputs": formatted_preds,
                "predictions": gathered_predictions,
                "image_ids": image_ids,
                "batch_size": batch_size,
            }

    def _format_predictions(self, predictions, image_idxs):
        """
        Formats the predictions outputted by faster-RCNN, a list of dictionaries containing
        'boxes', 'labels' and 'scores' into a format that is suitable for gathering across processes without
        losing the association between the image and annotations.

        The predictions are formated into an [N, 7] tensor with columns
        [x1, y1, x2, y2, score, class_label, image_id]
        """

        annotations_list = []
        for pred, image_idx in zip(predictions, image_idxs):
            for box, score, label in zip(pred["boxes"], pred["scores"], pred["labels"]):
                annotation = torch.cat(
                    (
                        box,
                        score[None],
                        label[None],
                        image_idx[None],
                    )
                )
                annotations_list.append(annotation)

        if not annotations_list:
            # create a 'padding' tensor so we can gather in all processes
            annotations = torch.tensor(
                [FRCNN_PADDING_VALUE] * 7,
                device=self.device,
            )
        else:
            annotations = torch.stack(annotations_list)

        return annotations

    def eval_epoch_end(self):
        self._add_additional_losses_to_run_history()

    def _add_additional_losses_to_run_history(self):
        for loss_name in self._additional_losses:
            self.run_history.update_metric(
                f"{self._metric_prefix}{loss_name}",
                self._additional_loss_trackers[loss_name].average,
            )


def faster_rcnn_collate_fn(batch):
    """We need to make sure we have a list of images and list of targets instead of list of tuples

    The way dataset works, it outputs one tuple per sample
        (image_tensor, target_dict)
    Therefore, each batch is originally a List[Tuple]
    This functions reformats them to have (as model needs)
        Tuple[Tensor[image_tensor], List[target_dict]] per batch
    """
    images, targets, image_idxs = zip(*batch)
    image_idxs = torch.stack(image_idxs).int()
    images = torch.stack(images).float()
    targets = list(targets)
    return images, targets, image_idxs
