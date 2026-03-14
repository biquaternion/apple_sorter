#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torchvision.transforms.v2 as T
from PIL import Image
import torchvision


class EfficientNetAppleClassifier:

    def __init__(self,
                 num_classes,
                 weights_path,
                 device="cpu",
                 input_size=224,
                 threshold=0.5):

        self.device = torch.device(device)
        self.threshold = threshold

        self.model = torchvision.models.efficientnet_b0(pretrained=False)
        self.model.classifier[1] = torch.nn.Linear(1280, num_classes)

        checkpoint = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        self.model.to(self.device)
        self.model.eval()

        self.transform = T.Compose([T.Resize((input_size, input_size)),
                                    T.ToImage(),
                                    T.ToDtype(torch.float32, scale=True),
                                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])])

    @staticmethod
    def _crop(image, bbox):
        h, w, _ = image.shape
        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        return image[y1:y2, x1:x2]

    def predict(self, crop):
        img = Image.fromarray(crop[:, :, ::-1])
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
        return probs[0, 1].item()

    def filter(self, image, detections):
        kept, removed = [], []
        for det in detections:
            crop = self._crop(image, det['bbox'])
            prob = self.predict(crop)
            det['cls_score'] = prob
            if prob >= self.threshold:
                kept.append(det)
            else:
                removed.append(det)
        return kept, removed