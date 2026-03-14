#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict


class BaseDetector(ABC):

    @abstractmethod
    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Returns list of detections:
        [
            {
                "bbox": [x1, y1, x2, y2],
                "score": float,
                "label": str
            }
        ]
        """
        pass