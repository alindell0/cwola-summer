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
    gd1_stars = gd1_stars.drop_duplicates(ignore_index=True)

    print(f'Total number of unique CWoLa top stars = {len(top_stars)}')
    print(f'Total number of true stream stars of the unique CWoLa top stars = {len(top_stream_stars)}/{len(top_stars)} ({len(top_stream_stars)/len(top_stars)*100:.2f}%)')

# plotting, etc
# compare final top stars non gd-1 after euclidean distance cutoff to those from paper? or reproduced with og pm param


## comparison with rotpmra

