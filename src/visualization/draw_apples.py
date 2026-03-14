#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2


def draw_ordered_apples(image, apples):
    canvas = image.copy()
    for order, apple in enumerate(apples, start=1):
        x1, y1, x2, y2 = apple['bbox']
        depth = apple.get('depth', None)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 3)

        text = f'{order}'
        if depth is not None:
            text += f' d={depth:.2f}'
        text_pos = (x1, max(0, y1 - 10))
        cv2.putText(canvas, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 255), 2, cv2.LINE_AA)

    return canvas
