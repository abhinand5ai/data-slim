import torch
import numpy as np
from utils import logger, utils
import argparse
import os

DEVICE = torch.device(str(os.environ.get("DEVICE", "cpu")))
NUM_GPUS = int(os.environ.get("NUM_GPUS", 0))


def get_data(top_rights, bottom_lefts, times, output_path):
    print("Getting data...")
    pass

def get_desired_filenames(filenames, times):
    desired_filenames = []
    for filename in filenames:
        if filename.find("mask") != -1:
            mask_filenames = filename
            continue
        file_time_idx = filename.rpartition(".")[0].split("_")[-1]
        file_time_idx = int(file_time_idx)
        if file_time_idx >= times[0] and file_time_idx <= times[1]:
            desired_filenames.append(filename)
            
    desired_filenames.append(mask_filenames)
    return desired_filenames


if __name__ == "__main__":
    get_data()
    