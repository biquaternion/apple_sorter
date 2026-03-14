#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import sys
from io import StringIO
from pathlib import Path

import cv2
import pandas as pd

from logging import getLogger

from utils.logging_config import setup_logging
from visualization.draw_apples import draw_ordered_apples

logger = getLogger(__name__)

def read_stdin():
    input_text = sys.stdin.read()
    if not input_text.strip():
        logger.error('no input received from stdin')
        return None
    logger.info('received piped input')
    logger.info(input_text)
    csv_parts = input_text.split('---\n') if '---\n' in input_text else [input_text]
    # data_columns = ['image', 'bbox', 'depth', 'conf', 'label', 'cls_conf']
    data_columns = ['image', 'center', 'depth']
    df_full = pd.DataFrame(columns=data_columns)
    for i, csv_part in enumerate(csv_parts):
        stripped = csv_part.strip()
        if stripped:
            logger.info(f'processing csv part {i + 1} / {len(csv_parts)} :')
            try:
                df = pd.read_csv(StringIO(csv_part), header=None, names=data_columns)
                df_full = pd.concat([df_full, df], ignore_index=True)
            except pd.errors.EmptyDataError as e:
                logger.error(f'error reading csv part {i + 1} / {len(csv_parts)} : {e}')
                continue
            except pd.errors.ParserError as e:
                logger.error(f'error parsing csv part {i + 1} / {len(csv_parts)} : {e}')
                continue
            except Exception as e:
                logger.error(f'unexpected error parsing csv part {i + 1} / {len(csv_parts)} : {e}')
                continue
    return df_full

if __name__ == '__main__':
    setup_logging('.')
    dataframe = read_stdin()
    print(dataframe)
    if dataframe is None:
        sys.exit(1)
    for k, im_df in dataframe.groupby('image'):
        apples = im_df.to_dict(orient='records')
        apples = [{'center': [int(x) for x in json.loads(apple['center'])],
                  'depth': apple['depth']} for apple in apples]
        out_im = draw_ordered_apples(cv2.imread(im_df['image'].iloc[0]), apples)
        cv2.imshow(Path(im_df['image'].iloc[0]).name, out_im)
        cv2.waitKey()
