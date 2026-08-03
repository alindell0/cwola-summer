from functions import *
import pandas as pd

## reproducibility with rotpmdec and comparison with rotpmra

scan_vars = ['rotpmdec', 'rotpmra']

for i in range(len(scan_vars)):
    scan_var = scan_vars[i]

    top_dfs = []
    gd1stars_dfs = []
    for idx in tqdm(range(21), desc='Patch'):
      train_on_patch(patch_idx=idx, scan_var=scan_var, fid_cuts=True)
    
      df_test = pd.read_hdf(f'..results/training/patch_{idx}/{scan_var}/df_test.h5')
      df_gd1stars = df_test[df_test['stream']==True]
      gd1stars_dfs.append(df_gd1stars)
      
      df_top = pd.read_hdf(f'../results/patch_{idx}/{scan_var}/df_top.h5')
      top_dfs.append(df_top)
    
      print(f'Successfully loaded df_gd1 and df_top for Patch {idx} and scan variable {scan_var}')
    
    top_stars = pd.concat([df for df in top_dfs])
    gd1_stars = pd.concat([df for df in gd1stars_dfs])
    
    top_stars = top_stars.drop_duplicates(ignore_index=True)
    top_stream_stars = top_stars[top_stars['stream']==True]
    top_background_stars = top_stars[top_stars['stream']==False]
    gd1_stars = gd1_stars.drop_duplicates(ignore_index=True)

    print(f'Total number of unique CWoLa top stars = {len(top_stars)}')
    print(f'Total number of true stream stars of the unique CWoLa top stars = {len(top_stream_stars)}/{len(top_stars)} ({len(top_stream_stars)/len(top_stars)*100:.2f}%)')

    save_folder = f'../results/fullgd1/{scan_var}/raw'
    
    plt.figure()
    plt.scatter(gd1_stars['ra'], gd1_stars['dec'], label='GD-1 Stars', color='grey', marker='.', s=5)
    plt.scatter(top_stream_stars['ra'], top_stream_stars['dec'], label='CWoLa Matches', color='red', marker='.', s=5)
    plt.scatter(top_background_stars['ra'], top_background_stars['dec'], label='CWoLa Non-Matches', color='blue', marker='.', s=5)
    plt.set_xlabel('Right Ascension α [°]', fontsize = 10)
    plt.set_ylabel('Declination δ [°]', fontsize = 10)
    plt.legend()
    plt.savefig(os.path.join(save_folder, "position.png"))
    plt.close()
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), tight_layout=True)    
    axes[0].scatter(gd1_stars['rotpmra'], gd1_stars['rotpmdec'], label='GD-1 Stars', color='grey', marker='.', s=5)
    axes[0].scatter(top_stream_stars['rotpmra'], top_stream_stars['rotpmdec'], label='CWoLa Matches', color='red', marker='.', s=5)
    axes[0].scatter(top_background_stars['rotpmra'], top_background_stars['rotpmdec'], label='CWoLa Non-Matches', color='blue', marker='.', s=5)
    axes[0].set_xlabel('Rotated Proper Motion (Right Ascension) μ_ϕcosλ [mas/yr]', fontsize = 10)
    axes[0].set_ylabel('Rotated Proper Motion (Declination) μ_λ [mas/yr]', fontsize = 10)

    axes[1].scatter(gd1_stars['b-r'], gd1_stars['g'], label='GD-1 Stars', color='grey', marker='.', s=5)
    axes[1].scatter(top_stream_stars['b-r'], top_stream_stars['g'], label='CWoLa Matches', color='red', marker='.', s=5)
    axes[1].scatter(top_background_stars['b-r'], top_background_stars['g'], label='CWoLa Non-Matches', color='blue', marker='.', s=5)
    axes[1].set_xlabel('Color b-r', fontsize = 10)
    axes[1].set_ylabel('Magnitude g', fontsize = 10)

    fig.savefig(os.path.join(save_folder, "pm_photometric.png"))
    plt.close()

    ## kmeans here !!!!!

    save_folder = f'../results/fullgd1/{scan_var}/k_means'


# compare final top stars non gd-1 after euclidean distance cutoff to those from paper? or reproduced with og pm param


## comparison with rotpmra

