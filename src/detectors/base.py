#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict


class BaseDetector(ABC):

    @abstractmethod
    def detect(self, image: np.ndarray) -> List[Dict]:
        pass