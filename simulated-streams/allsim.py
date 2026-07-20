import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
import wandb
import os
import seaborn as sns
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append('../scripts')
from functions import *

df = pd.read_hdf('simulated-data/simulated-streams100.h5')

df_100sims = {name: group for name, group in df.groupby('file')}

for idx in range(100):
    df = df_100sims[idx]

    plot_data(df, save_folder = f'cwola/plots/sim-patch{idx}')
    df_regions = signal_sideband(df, pm_parameter='rotpmdec', sig_factor=0.25, sb_factor=0.5, bin_num=55)
    df_test_full = cwola_train(df, pm_parameter='rotpmdec', dropout=0.2, k_folds=5, batch_size=10000, lr=0.001, patience=30, epochs=100, trainval_loops=3, save_folder=f'cwola/results/sim-patch{idx}', wandbproj=f'sim-patch{idx}')
    get_results(df_test_full, top_n=250, save_folder=f'cwola/results/sim-patch{idx}/plots')

    
    

