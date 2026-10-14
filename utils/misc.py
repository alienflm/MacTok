import torch
import os
import numpy as np
import random

import datetime
import functools
import glob
import os
import subprocess
import sys
import time
from collections import defaultdict, deque
from typing import Iterator, List, Tuple

import numpy as np
import torch
import torch.distributed as tdist

import argparse



def str2bool(v):
    """
    str to bool
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
    

def manage_checkpoints(save_dir, keep_last_n=15):
    # List all checkpoint files in the save directory
    checkpoints = [f for f in os.listdir(save_dir) if f.endswith('.pt')]
    checkpoints = [f for f in checkpoints if 'best_ckpt' not in f]
    checkpoints.sort(key=lambda f: int(f.split('/')[-1].split('.')[0]))  # Sort by epoch number

    # If more than `keep_last_n` checkpoints exist, remove the oldest ones
    if len(checkpoints) > keep_last_n + 1:  # keep_last_n + 1 to account for the latest checkpoint
        for checkpoint_file in checkpoints[:-keep_last_n-1]:
            checkpoint_path = os.path.join(save_dir, checkpoint_file)
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
                print(f"Removed old checkpoint: {checkpoint_path}")


def load_model_state_dict(orig_state_dict):
    model_state = {}
    for key, value in orig_state_dict.items():
        if key.startswith("module."):
            model_state[key[7:]] = value
        if key.startswith("_orig_mod."):
            model_state[key[10:]] = value
        else:
            model_state[key] = value
    return model_state