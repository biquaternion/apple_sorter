#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Optional

import numpy as np
import logging

from pipeline.postprocessing import filter_by_label, filter_by_box_nesting

logger = logging.getLogger(__name__)

class ApplePipeline:
    def __init__(self, detector, depth, classifier: Optional = None):
        logger.info('Initialization')
        self.detector = detector
        self.depth = depth
        self.classifier = classifier

    def run(self, image):
        detections = self.detector.detect(image)

        logger.info('Filtering by label:')
        before = len(detections)
        detections, filtered_by_label = filter_by_label(detections, 'apple')
        logger.info(f'\tbefore: {before}\t;\tafter: {len(detections)}')

        logger.info(f'filtered by nesting:')
        before = len(detections)
        detections, filtered_by_nesting = filter_by_box_nesting(detections, return_inner=True)
        logger.info(f'\tbefore: {before}\t;\tafter: {len(detections)}')

        if self.classifier is not None:
            logger.info('Filtering via classifier:')
            before = len(detections)
            detections, filtered_by_classifier = self.classifier.filter(image, detections)
            logger.info(f'\tbefore: {before}\t;\tafter: {len(detections)}')
        else:
            logger.warning('No classifier provided')

        depth_map = self.depth.predict(image)

        apples = []

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            region = depth_map[y1:y2, x1:x2]
            x = (x1 + x2) // 2
            y = (y1 + y2) // 2
            center = [x, y]
            depth_cands = [np.percentile(region, 30), depth_map[y, x], np.median(region)]
            depth_value = np.median(depth_cands)
            apples.append({**det, 'depth': float(depth_value), 'center': center})
        apples_sorted = sorted(apples, key=lambda x: x['depth'], reverse=True)

        return apples_sorted
