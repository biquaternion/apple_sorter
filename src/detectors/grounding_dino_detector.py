#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
import cv2

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from .base import BaseDetector


class GroundingDINODetector(BaseDetector):

    def __init__(self, model_name: str = 'IDEA-Research/grounding-dino-base',
                 text_prompt: str = 'apple', threshold: float = 0.3,
                 device: str = 'cuda'):
        # self.device = torch.device(device if torch.cuda.is_available() else 'mps')
        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name)
        self.model.to(self.device).eval()
        self.text_prompt = text_prompt
        self.threshold = threshold

    @torch.no_grad()
    def detect(self, image: np.ndarray):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=image,
                                text=self.text_prompt,
                                return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(outputs,
                                                                        threshold=self.threshold,
                                                                        target_sizes=[image.shape[:2]])[0]
        detections = []
        for box, score, label in zip(results['boxes'], results['scores'], results['labels']):
            x1, y1, x2, y2 = box.cpu().numpy()
            detections.append({'bbox': [int(x1), int(y1), int(x2), int(y2)],
                               'score': float(score),
                               'label': str(label)})
        return detections
