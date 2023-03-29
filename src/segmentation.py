import os

import numpy as np
import torch
import torchvision
from torchvision.models import feature_extraction

from Maskrcnn.network_files import MaskRCNN, AnchorsGenerator
from Maskrcnn.backbone import BackboneWithFPN, LastLevelMaxPool


class Segmentation:
    def __init__(self,device):
        self.device = device

        num_classes = 2  # 不包含背景
        box_thresh = 0.8
        weights_path = "src/grcnn/models/model_19.pth"

        backbone = torchvision.models.mobilenet_v3_large()
        return_layers = {
            "features.6": "0",
            "features.12": "1",
            "features.16": "2",
        }
        in_channels_list = [40, 112, 960]
        new_backbone = feature_extraction.create_feature_extractor(
            backbone, return_layers
        )

        backbone_with_fpn = BackboneWithFPN(
            new_backbone,
            return_layers=return_layers,
            in_channels_list=in_channels_list,
            out_channels=256,
            extra_blocks=LastLevelMaxPool(),
            re_getter=False,
        )

        anchor_sizes = ((64,), (128,), (256,), (512,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        anchor_generator = AnchorsGenerator(
            sizes=anchor_sizes, aspect_ratios=aspect_ratios
        )

        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=["0", "1", "2"], output_size=[7, 7], sampling_ratio=2
        )

        self.model = MaskRCNN(
            backbone=backbone_with_fpn,
            num_classes=num_classes + 1,
            rpn_score_thresh=box_thresh,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
            box_score_thresh=box_thresh,
        )

        assert os.path.exists(weights_path), "{} file dose not exist.".format(
            weights_path
        )
        weights_dict = torch.load(weights_path, map_location="cuda:0")
        weights_dict = (
            weights_dict["model"] if "model" in weights_dict else weights_dict
        )
        self.model.load_state_dict(weights_dict)
        self.model.to(self.device)
        self.model.eval()

        unlazy = torch.zeros((1, 3, 500, 500)).to(self.device)
        self.model(unlazy)

    def __call__(self, original_img):
        img = torch.from_numpy(original_img).to(self.device)

        with torch.no_grad():
            img = torch.unsqueeze(img, dim=0)  # [1, 3, 480, 640]
            predictions = self.model(img)[0]

            predict_boxes = predictions["boxes"].to("cpu").numpy()
            predict_classes = predictions["labels"].to("cpu").numpy()
            predict_scores = predictions["scores"].to("cpu").numpy()
            predict_masks = predictions["masks"].to("cpu").numpy()
            predict_masks = np.squeeze(
                predict_masks, axis=1
            )  # [batch, 1, h, w] -> [batch, h, w]

            if len(predict_boxes) == 0:
                raise Exception("分割无结果")

            return predict_boxes, predict_classes, predict_scores, predict_masks
