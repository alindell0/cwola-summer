import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import sys
# sys.path.append('../scripts')
from checkfunctions import *

import seaborn as sns
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import copy
import wandb
import os

## distributed setup — torchrun sets these env vars, one process per GPU.
## training is split across ranks by test fold (embarrassingly parallel, no
## DistributedDataParallel/gradient sync needed), so each process just needs
## to know which GPU is "its own".
rank = int(os.environ.get("RANK", 0))
local_rank = int(os.environ.get("LOCAL_RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))
torch.cuda.set_device(local_rank)
device = torch.device(f'cuda:{local_rank}')
print(f'[rank {rank}/{world_size}] using {device}')

## load data
df = pd.read_hdf('./simulated-data/simulated_patch.h5')
df.columns = ["pmdec", "pmra", "dec", "ra", "b-r", "g",
                "rotra", "rotdec", "rotpmra", "rotpmdec", "stream", "ra_wrapped"]

## plot data — only rank 0, so 4 processes don't race on the same PNG files
plot_data(df, save_folder = './checkplots', verbose=(rank == 0))

## region labels and plots (every rank needs df_regions to build its folds)
df_regions = signal_sideband(df, bin_num=55, verbose=(rank == 0))

## train — each rank trains/tests only its own subset of the 5 test folds
df_test_partial = cwola_train(
    df_regions,
    pm_parameter='rotpmdec',
    dropout=0.2,
    k_folds=5,
    batch_size=10000,
    lr=0.001,
    patience=30,
    epochs=100,
    trainval_loops=3,
    save_folder='./results/checksim1',
    wandbproj='checksim1',
    device=device,
    rank=rank,
    world_size=world_size,
)

## NOTE: final merge across ranks + get_results() plotting happens once, after
## torchrun returns, in merge_results.py — not here.
