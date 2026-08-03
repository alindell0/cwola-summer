from functions import *
import pandas as pd
from tqdm import tqdm

### full training function
def train_on_patch(patch_idx, scan_var, fid_cuts):
    print(f'Training Patch {patch_idx} with signal parameter {scan_var}...')
    df = load_data(patch_idx)
    df['patch'] = patch_idx
    plot_data(df, save_folder = f'../gaia-data/plots/patch_{patch_idx}')
    df_regions = signal_sideband(df, save_folder = f'../results/patch_{patch_idx}/{scan_var}', 
                               pm_parameter=scan_var, sig_factor=1, sb_factor=3)
    df_test_full = cwola_train(df, pm_parameter=scan_var, dropout=0.2, k_folds=5, batch_size=10000, 
                             lr=0.001, patience=30, epochs=100, trainval_loops=3, 
                             save_folder=f'../results/patch_{patch_idx}/training', wandbproj=None)
    get_results(df_test_full, top_n=250, fid_cuts=fid_cuts, save_folder=f'../results/patch_{patch_idx}', patch_idx=patch_idx)


## reproducibility with rotpmdec, ## comparison with rotpmra

### for gpu ???
# rewrite idx = SLURM_ARRAY_TASK_ID 
# array=0-20 in job submission

top_dfs = []
gd1stars_dfs = []
for idx in tqdm(range(21), desc='Patch'):
  train_on_patch(patch_idx=idx, scan_var='rotpmra', fid_cuts=True)

  df_test = pd.read_hdf(f'..results/training/patch_{idx}/df_test.h5')
  df_gd1stars = df_test[df_test['stream']==True]
  gd1stars_dfs.append(df_gd1stars)
  
  df_top = pd.read_hdf(f'../results/patch_{idx}/df_top.h5')
  top_dfs.append(df_top)

  print(f'Successfully loaded df_gd1 and df_top for Patch {idx}')


top_stars = pd.concat([df for df in top_dfs])
gd1_stars = pd.concat([df for df in gd1stars_dfs])

# drop duplicates, plotting, etc

# compare final top stars non gd-1 after euclidean distance cutoff to those from paper? or reproduced with og pm param

