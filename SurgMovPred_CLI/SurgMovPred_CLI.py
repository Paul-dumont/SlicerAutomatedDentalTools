#!/usr/bin/env python-real

import sys, argparse, os, traceback, glob, json
from pathlib import Path

import sys
import logging

# ===== Logging Configuration =====
logger = logging.getLogger("SurgMovPred_CLI")
logger.setLevel(logging.INFO)
logger.propagate = False
if logger.handlers:
    logger.handlers.clear()
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)s - %(levelname)s - (%(filename)s:%(lineno)d) - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

logger.info("SurgMovPred_CLI.py run")

def main(args):
    logger.info(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('inputFolder', type=str)
    parser.add_argument('modelPath', type=str)
    parser.add_argument("outputFolder", type=str)

    args = parser.parse_args()
    main(args)