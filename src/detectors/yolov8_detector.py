#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ultralytics import YOLO
from .base import BaseDetector


class YOLOv8Detector(BaseDetector):

    def __init__(self, model_path: str, conf: float = 0.25):
        self.model = YOLO(model_path, verbose=False)
        self.conf = conf

    def detect(self, image):
        results = self.model(image,
                             conf=self.conf,
                             verbose=False)[0]

        detections = []

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            score = float(box.conf)
            cls = int(box.cls)

            detections.append(
                {
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "score": score,
                    "label": self.model.names[cls],
                }
            )

        return detections
