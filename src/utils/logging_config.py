#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging.config
from pathlib import Path


def setup_logging(log_dir: str):

    Path(log_dir).mkdir(parents=True, exist_ok=True)

    config = {
        "version": 1,
        "disable_existing_loggers": False,

        "formatters": {

            "console": {
                "format": "[%(levelname)s] %(message)s"
            },

            "file": {
                "format": "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },

            "debug_file": {
                "format": "%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s"
            },
        },

        "handlers": {

            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "console",
                "stream": "ext://sys.stderr",
            },

            "file": {
                "class": "logging.FileHandler",
                "level": "INFO",
                "formatter": "file",
                "filename": f"{log_dir}/run.log",
            },

            "debug_file": {
                "class": "logging.FileHandler",
                "level": "DEBUG",
                "formatter": "debug_file",
                "filename": f"{log_dir}/debug.log",
            },
        },

        "root": {
            "level": "DEBUG",
            "handlers": ["console", "file", "debug_file"],
        },
    }

    logging.config.dictConfig(config)