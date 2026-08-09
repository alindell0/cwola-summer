from functions import *


def train_on_patch(patch_idx, scan_var, fid_cuts):
    print(f'Training Patch {patch_idx} with signal parameter {scan_var}...')
    df = load_data(patch_idx)
    df['patch'] = patch_idx
    plot_data(df, save_folder = f'../gaia-data/plots/patch_{patch_idx}')
    df_regions = signal_sideband(df, save_folder = f'../results/patch_{patch_idx}/{scan_var}/epochs50', 
                               pm_parameter=scan_var, sig_factor=1, sb_factor=3)
    df_test_full = cwola_train(df_regions, pm_parameter=scan_var, dropout=0.2, k_folds=5, batch_size=10000,
                             lr=0.001, patience=30, epochs=50, trainval_loops=3, 
                             save_folder=f'../results/patch_{patch_idx}/{scan_var}/epochs50/training', wandbproj=None)
    get_results(df_test_full, top_n=250, fid_cuts=fid_cuts, save_folder=f'../results/patch_{patch_idx}/{scan_var}/epochs50', patch_idx=patch_idx)

print('Training Patch 0 with signal parameter rotpmra with 50 epochs to check purity...')
train_on_patch(patch_idx=0, scan_var='rotpmra', fid_cuts=True)

print('All done!')