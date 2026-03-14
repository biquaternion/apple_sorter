#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
import numpy as np


class BaseDepthEstimator(ABC):

    @abstractmethod
    def predict(self, image: np.ndarray) -> np.ndarray:
        pass