import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import sys
# sys.path.append('../scripts')
from functions import *

import seaborn as sns
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import copy
import wandb
import os

## load data
df = pd.read_hdf('./simulated-data/simulated_patch.h5')
df.columns = ["pmdec", "pmra", "dec", "ra", "b-r", "g", 
                "rotra", "rotdec", "rotpmra", "rotpmdec", "stream", "ra_wrapped"]

## plot data
plot_data(df, save_folder = './checkplots')

## region labels and plots
df_regions = signal_sideband(df, bin_num=55)

## train
df_test_full = cwola_train(
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
    wandbproj='checksim1'
)

## plot results
get_results(df_test_full, top_n=250, save_folder='./results/checksim1/plots')
