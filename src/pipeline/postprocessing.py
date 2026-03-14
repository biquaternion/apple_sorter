#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from itertools import combinations


def filter_by_label(detections, label):
    approved = [d for d in detections if label in d['label']]
    filtered = [d for d in detections if label not in d['label']]
    return approved, filtered


def filter_by_box_nesting(detections, return_inner=True, tol=0.03):
    remove = set()

    for i, j in combinations(range(len(detections)), 2):
        a = detections[i]['bbox']
        b = detections[j]['bbox']

        aw, ah = a[2] - a[0], a[3] - a[1]
        bw, bh = b[2] - b[0], b[3] - b[1]

        tol_ax, tol_ay = aw * tol, ah * tol
        tol_bx, tol_by = bw * tol, bh * tol

        a_in_b = (
                a[0] >= b[0] - tol_bx and
                a[1] >= b[1] - tol_by and
                a[2] <= b[2] + tol_bx and
                a[3] <= b[3] + tol_by
        )

        b_in_a = (b[0] >= (a[0] - tol_ax) and b[1] >= (a[1] - tol_ay) and
                  b[2] <= (a[2] + tol_ax) and b[3] <= (a[3] + tol_ay))

        if a_in_b and not b_in_a:
            remove.add(j if return_inner else i)
        elif b_in_a and not a_in_b:
            remove.add(i if return_inner else j)

    kept = [d for k, d in enumerate(detections) if k not in remove]
    removed = [d for k, d in enumerate(detections) if k in remove]

    return kept, removed


if __name__ == '__main__':
    pass
