### script for patch 0, comparing proper motions variables, comparing fiducial cuts

import torch

import sys
sys.path.append('../scripts')
from functions import *


df_patch0 = pd.read_hdf('gd1_patch0.h5')
plot_data(df_patch0, save_folder='../patch0-hpresults/data/no_fid_cuts')

df_patch0_fidcuts = fiducial_cuts(df_patch0)
plot_data(df_patch0_fidcuts, save_folder='../patch0-hpresults/data/fid_cuts')


# pm declination
df_regions = signal_sideband(df_patch0, save_folder = '../patch0-hpresults/pmdec', pm_parameter='rotpmdec', 
                             sig_factor=1, sb_factor=3, bin_num=50, verbose=True)
df_test = cwola_train(df_regions, pm_parameter='rotpmdec', dropout=0.2, k_folds=5, batch_size=10000, lr=0.001, 
                      patience=30, epochs=100, trainval_loops=3, save_folder='../patch0-hpresults/pmdec', 
                      wandbproj=None, device=None, rank=0)
## with fiducial cuts
fid_results = get_results(df_test, top_n=250, fid_cuts=True save_folder='../patch0-hpresults/pmdec/fid_cuts')
## without fiducial cuts
no_fid_results = get_results(df_test, top_n=250, fid_cuts=False save_folder='../patch0-hpresults/pmdec/no_fid_cuts')



# pm right ascension
df_regions = signal_sideband(df_patch0, save_folder = '../patch0-hpresults/pmra', pm_parameter='rotpmra', 
                             sig_factor=1, sb_factor=3, bin_num=50, verbose=True)
df_test = cwola_train(df_regions, pm_parameter='rotpmra', dropout=0.2, k_folds=5, batch_size=10000, lr=0.001, 
                      patience=30, epochs=100, trainval_loops=3, save_folder='../patch0-hpresults/pmra', 
                      wandbproj=None, device=None, rank=0)
## with fiducial cuts
fid_results = get_results(df_test, top_n=250, fid_cuts=True save_folder='../patch0-hpresults/pmra/fid_cuts')
## without fiducial cuts
no_fid_results = get_results(df_test, top_n=250, fid_cuts=False save_folder='../patch0-hpresults/pmra/no_fid_cuts')



