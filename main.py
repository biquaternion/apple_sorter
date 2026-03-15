#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from itertools import chain
from pathlib import Path

import hydra
import cv2
import pandas as pd

from hydra.utils import instantiate
from src.visualization.draw_apples import draw_ordered_apples
from src.utils.logging_config import setup_logging
import logging
from huggingface_hub.utils import logging as hf_logging
hf_logging.set_verbosity_error()
logging.getLogger("httpx").setLevel(logging.ERROR)

tkinter_imported = None

try:
    from tkinter import Tk
    from tkinter.filedialog import askopenfilenames
    tkinter_imported = True
except ImportError:
    tkinter_imported = False

@hydra.main(config_path='configs', config_name='config', version_base=None)
def main(cfg):

    setup_logging(cfg.logging.dir)
    logger = logging.getLogger(__name__)

    pipeline = instantiate(cfg.pipeline)

    extensions = ['*.jpg', '*.jpeg', '*.png']
    extensions = extensions + list(map(lambda x: x.upper(), extensions))
    logger.info(f'supported image extensions: {extensions}')

    input_path = Path(cfg.input_path)
    logger.debug(f'Input path: {input_path.absolute()}, {input_path.exists()}')
    image_paths = chain.from_iterable(input_path.glob(ext) for ext in extensions) if input_path.is_dir() else [input_path]
    if cfg.interactive:
        if tkinter_imported:
            Tk().withdraw()
            image_paths = map(Path, askopenfilenames(initialdir=input_path.parent))
        else:
            logger.warning('Interactive mode requires tkinter, which is not available. Using default image path.')
    image_paths = list(image_paths)
    logger.info(f'{len(image_paths)} images to process')
    output_dir_path = Path(cfg.output_path)
    output_columns = ['image', 'center', 'depth']
    output_df = pd.DataFrame(columns=output_columns)
    for im_p in image_paths:
        logger.info(f'Processing {im_p}')
        image = cv2.imread(str(im_p))

        logger.info(f'Running pipeline on {im_p}')
        apples_data = pipeline.run(image)

        logger.info(apples_data)
        if apples_data:
            df = pd.DataFrame(apples_data)
        else:
            df = pd.DataFrame(columns=output_columns)  # empty if no apples found
        output_image = draw_ordered_apples(image, apples_data)

        output_path = output_dir_path / im_p.name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df['image'] = str(im_p.absolute().resolve())
        # single_output = df[output_columns].to_csv(index=False)
        written = cv2.imwrite(str(output_path), output_image)
        if not written:
            logger.error(f'Error writing image to {output_path}')
        else:
            logger.info(f'Written image to {output_path.absolute()}')
        output_df = pd.concat([output_df, df], ignore_index=True)
    output_path = output_dir_path / 'output.csv'
    output_df.to_csv(output_path, index=False)
    output = output_df[output_columns].to_csv(index=False, header=False)
    print(output)


if __name__ == '__main__':
    main()
