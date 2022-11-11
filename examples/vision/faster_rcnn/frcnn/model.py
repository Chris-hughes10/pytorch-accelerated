from torch import nn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import (
    FastRCNNPredictor,
    fasterrcnn_resnet50_fpn,
)


def create_frcnn_model(num_classes: int, image_size=800):
    """Return a pretrained Faster-RCNN model with a ResNet-50 backbone

    The backbone is trained on Imagnet and the head on COCO

    IMPORTANT: This model crops and normalizes images inside, so no need to do that before.
        It will make sure INSIDE the model that images are the right size and normalized.

    :param num_classes: Number of classes in target (excluding background class)
    """

    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

    model = fasterrcnn_resnet50_fpn(
        min_size=image_size,
        box_detections_per_img=100,
        box_batch_size_per_image=512,
        rpn_anchor_generator=rpn_anchor_generator,
    )
    # Number of features that enter the classifier head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained classifier head with a new one to match our classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    return model
