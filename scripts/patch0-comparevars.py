import torch
%pip install tqdm

from functions import *

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
                             save_folder=f'../results/patch_{patch_idx}/{scan_var}/training', wandbproj=None)
    get_results(df_test_full, top_n=250, fid_cuts=fid_cuts, save_folder=f'../results/patch_{patch_idx}/{scan_var}', patch_idx=patch_idx)


scan_vars = ['rotpmdec', 'rotpmra']

for i in range(len(scan_vars)):
    scan_var = scan_vars[i]
    train_on_patch(patch_idx=0, scan_var=scan_var, fid_cuts=True)

  
  
