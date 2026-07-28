
import torch

import sys
sys.path.append('../scripts')
from functions import *


df_patch0 = pd.read_hdf('gd1_patch0.h5')

plot_data(df_patch0, save_folder = None, verbose=True)

df_regions = signal_sideband(df_patch0, pm_parameter='rotpmdec', sig_factor=1, sb_factor=3, bin_num=71, verbose=True)

df_test = cwola_train(df_regions, 
                      pm_parameter='rotpmdec', 
                      dropout=0.2, 
                      k_folds=5, 
                      batch_size=10000, 
                      lr=0.001, 
                      patience=30, 
                      epochs=100, 
                      trainval_loops=3, 
                      save_folder='results/checkpatch0',
                      wandbproj='checkpatch0', 
                      device=None, 
                      rank=0)