#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from enum import Enum

import cv2


class DrawMode(Enum):
    CENTER = 1
    BOX = 2
    BOTH = 3


def draw_apple_center(canvas, x, y):
    h, w, _ = canvas.shape
    radius = h // 360
    cv2.circle(canvas, (x, y), radius * 3, (0, 0, 255), -1)
    cv2.circle(canvas, (x, y), radius * 2, (0, 255, 0), -1)
    cv2.circle(canvas, (x, y), radius, (255, 0, 0), -1)
    text_pos = (x, max(0, y - radius * 3 - 10))
    return text_pos

def draw_apple_box(canvas, x1, y1, x2, y2):
    h, w, _ = canvas.shape
    width = h // 360
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), width)
    text_pos = (x1, max(0, y1 - 10))
    return text_pos


def draw_ordered_apples(image, apples, mode: DrawMode=DrawMode.CENTER):
    canvas = image.copy()
    for order, apple in enumerate(apples, start=1):
        depth = apple.get('depth', None)
        if mode is DrawMode.CENTER:
            text_pos = draw_apple_center(canvas, *apple['center'])
        elif mode is DrawMode.BOX:
            text_pos = draw_apple_box(canvas, *apple['bbox'])
        elif mode is DrawMode.BOTH:
            draw_apple_center(canvas, *apple['center'])
            text_pos = draw_apple_box(canvas, *apple['bbox'])
        else:
            raise ValueError(f'Draw Mode {mode.name} not supported')

        text = f'{order}'
        if depth is not None:
            text += f' d={depth:.2f}'
        cv2.putText(canvas, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 255), 2, cv2.LINE_AA)

    return canvas
