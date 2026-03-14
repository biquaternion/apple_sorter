#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import torch

from transformers import (
    AutoImageProcessor,
    DepthAnythingForDepthEstimation,
)

from .base import BaseDepthEstimator


class DepthAnythingV2(BaseDepthEstimator):

    def __init__(
        self,
        model_name: str = "depth-anything/Depth-Anything-V2-Large-hf",
        device: str = "cuda",
    ):

        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu"
        )

        self.processor = AutoImageProcessor.from_pretrained(model_name)

        self.model = DepthAnythingForDepthEstimation.from_pretrained(
            model_name
        ).to(self.device)

        self.model.eval()

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> np.ndarray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        inputs = self.processor(images=image, return_tensors="pt")

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)

        depth = outputs.predicted_depth

        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        return depth.cpu().numpy()