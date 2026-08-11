from functions import *
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial import distance_matrix
import sys

sys.stdout = open("logs/gd1analysis.log", "a")
print('In the log file.')


scan_vars = ['rotpmdec','rotpmra']

for i in range(len(scan_vars)):
  scan_var = scan_vars[i]
  print(f'Analyzing full GD-1 for scan variable {scan_var}...')
  
  # combine results from test dfs from each patch
  top_dfs = [] 
  gd1stars_dfs = []
  for idx in tqdm(range(21), desc='Patch'):
    df_test = pd.read_hdf(f'../results/patch_{idx}/{scan_var}/training/df_test.h5')
    auc, ap = auc_ap_check(df_test)
    df_test = fiducial_cuts(df_test) # saved df_test.h5 was before fiducial cuts
    df_gd1stars = df_test[df_test['stream']==True]
    gd1stars_dfs.append(df_gd1stars)

    df_top = pd.read_hdf(f'../results/patch_{idx}/{scan_var}/fid_cuts/df_top.h5')
    top_dfs.append(df_top)

    print(f'Successfully loaded df_gd1 and df_top for Patch {idx} and scan variable {scan_var}')

  top_stars = pd.concat([df for df in top_dfs])
  gd1_stars = pd.concat([df for df in gd1stars_dfs])

  # adjust ra values to be between 0 and 360 for plotting purposes
  top_stars['ra_wrapped'] = top_stars['ra'].apply(lambda x: x if x > 100 else x + 360)
  gd1_stars['ra_wrapped'] = gd1_stars['ra'].apply(lambda x: x if x > 100 else x + 360)

  save_folder = f'../results/fullgd1/{scan_var}/raw'
  os.makedirs(save_folder, exist_ok=True)
  top_stars.to_hdf(os.path.join(save_folder, 'all_top_stars.h5'), key='all_top_stars', mode='w')

  # check differing neural network scores between duplicate stars
  check_cols = ['b-r', 'g', 'ra', 'dec', 'pmra', 'pmdec']
  duplicate_rows = top_stars[top_stars.duplicated(subset=check_cols, keep=False)].sort_values(by=check_cols)
  print(f'Duplicate rows found: {len(duplicate_rows)}')
  print(duplicate_rows)

  top_stars = top_stars.drop_duplicates(subset=check_cols,ignore_index=True)
  top_stars.to_hdf(os.path.join(save_folder, 'unique_top_stars.h5'), key='unique_top_stars', mode='w')
  top_stream_stars = top_stars[top_stars['stream']==True]
  top_background_stars = top_stars[top_stars['stream']==False]
  gd1_stars = gd1_stars.drop_duplicates(subset=check_cols,ignore_index=True)

  print(f'Before k-means clustering...')
  print(f'Total number of unique CWoLa top stars = {len(top_stars)}')
  print(f'Total number of GD-1 stars passing fiducial cuts = {len(gd1_stars)}')
  print(f'Total number of true stream stars of the unique CWoLa top stars = {len(top_stream_stars)}/{len(top_stars)} ({len(top_stream_stars)/len(top_stars)*100:.2f}%)')

  purity = len(top_stream_stars) / len(top_stars)
  print(f'Purity of the top cluster: {purity*100:.2f}% ({len(top_stream_stars)}/{len(top_stars)})')
  completeness = len(top_stream_stars) / len(gd1_stars)
  print(f'Completeness of the top cluster vs GD-1: {completeness*100:.2f}% ({len(top_stream_stars)}/{len(gd1_stars)})')


  # plot all top stars and gd1 stars in position, proper motion, and photometric space
  plt.figure()
  plt.scatter(gd1_stars['ra_wrapped'], gd1_stars['dec'], label='GD-1 Stars', color='grey', marker='.', s=5)
  plt.scatter(top_stream_stars['ra_wrapped'], top_stream_stars['dec'], label='CWoLa Matches', color='red', marker='.', s=5)
  plt.scatter(top_background_stars['ra_wrapped'], top_background_stars['dec'], label='CWoLa Non-Matches', color='blue', marker='.', s=5)
  plt.xlabel('Right Ascension α [°]', fontsize = 10)
  plt.ylabel('Declination δ [°]', fontsize = 10)
  plt.legend()
  plt.savefig(os.path.join(save_folder, "position.png"))
  plt.close()

  fig, axes = plt.subplots(1, 2, figsize=(10, 4), tight_layout=True)    
  axes[0].scatter(gd1_stars['rotpmra'], gd1_stars['rotpmdec'], label='GD-1 Stars', color='grey', marker='.', s=5)
  axes[0].scatter(top_stream_stars['rotpmra'], top_stream_stars['rotpmdec'], label='CWoLa Matches', color='red', marker='.', s=5)
  axes[0].scatter(top_background_stars['rotpmra'], top_background_stars['rotpmdec'], label='CWoLa Non-Matches', color='blue', marker='.', s=5)
  axes[0].set_xlabel('Rotated Proper Motion (Right Ascension) μ_ϕcosλ [mas/yr]', fontsize = 10)
  axes[0].set_ylabel('Rotated Proper Motion (Declination) μ_λ [mas/yr]', fontsize = 10)
  axes[0].legend()

  axes[1].scatter(gd1_stars['b-r'], gd1_stars['g'], label='GD-1 Stars', color='grey', marker='.', s=5)
  axes[1].scatter(top_stream_stars['b-r'], top_stream_stars['g'], label='CWoLa Matches', color='red', marker='.', s=5)
  axes[1].scatter(top_background_stars['b-r'], top_background_stars['g'], label='CWoLa Non-Matches', color='blue', marker='.', s=5)
  axes[1].set_xlabel('Color b-r', fontsize = 10)
  axes[1].set_ylabel('Magnitude g', fontsize = 10)
  axes[1].legend()

  fig.savefig(os.path.join(save_folder, "pm_photometric.png"))
  plt.close()


  save_folder = f'../results/fullgd1/{scan_var}/k_means'
  os.makedirs(save_folder, exist_ok=True)

  ## kmeans clustering here to separate the top stars into two groups based on their proper motions
  kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
  kmeans.fit(top_stars[['rotpmra', 'rotpmdec']])
  top_stars['kmeans_label'] = kmeans.labels_
  group_0 = top_stars[top_stars['kmeans_label'] == 0]
  print(f'Group 0: {len(group_0)} stars')
  group_1 = top_stars[top_stars['kmeans_label'] == 1]
  print(f'Group 1: {len(group_1)} stars')

  # plot clusters in proper motion space for visualization
  plt.scatter(group_0['rotpmra'], group_0['rotpmdec'], label='Group 0', marker='.', s=5)
  plt.scatter(group_1['rotpmra'], group_1['rotpmdec'], label='Group 1', marker='.', s=5)
  plt.xlabel('Rotated Proper Motion (Right Ascension) μ_ϕcosλ [mas/yr]', fontsize = 10)
  plt.ylabel('Rotated Proper Motion (Declination) μ_λ [mas/yr]', fontsize = 10)
  plt.title('K-Means Clustering of CWoLa Top Stars', fontsize = 12)
  plt.legend()
  plt.savefig(os.path.join(save_folder, "kmeans.png"))
  plt.close()

  if len(group_0) > len(group_1):
    top_cluster = group_0
    top_cluster_stream = top_cluster[top_cluster['stream']==True]
    if (len(top_cluster_stream)/len(top_cluster)) == 0:
        print('Warning: Top cluster has no stream stars. Using other cluster.')
        top_cluster = group_1
        print(f'Top cluster is Group 1 with {len(group_1)} stars')
    else:
       print(f'Top cluster is Group 0 with {len(group_0)} stars')
  if len(group_1) > len(group_0):
    top_cluster = group_1
    top_cluster_stream = top_cluster[top_cluster['stream']==True]
    if (len(top_cluster_stream)/len(top_cluster)) == 0:
        print('Warning: Top cluster has no stream stars. Using other cluster.')
        top_cluster = group_0
        print(f'Top cluster is Group 0 with {len(group_0)} stars')
    else:
       print(f'Top cluster is Group 1 with {len(group_1)} stars')


  top_cluster_stream = top_cluster[top_cluster['stream']==True]

  print(f'After k-means clustering...')
  print(f'Total number of unique CWoLa top stars = {len(top_cluster)}')
  print(f'Total number of GD-1 stars passing fiducial cuts = {len(gd1_stars)}')
  print(f'Total number of true stream stars of the unique CWoLa top stars = {len(top_cluster_stream)}/{len(top_cluster)} ({len(top_cluster_stream)/len(top_cluster)*100:.2f}%)')

  purity = len(top_cluster_stream) / len(top_cluster)
  print(f'Purity of the top cluster: {purity*100:.2f}% ({len(top_cluster_stream)}/{len(top_cluster)})')
  completeness = len(top_cluster_stream) / len(gd1_stars)
  print(f'Completeness of the top cluster vs GD-1: {completeness*100:.2f}% ({len(top_cluster_stream)}/{len(gd1_stars)})')


  # plot top cluster result over gd1 stars
  plt.figure()
  plt.scatter(gd1_stars['ra_wrapped'], gd1_stars['dec'], label='GD-1 Stars', color='grey', marker='.', s=5)
  plt.scatter(top_cluster['ra_wrapped'], top_cluster['dec'], label='CWoLa Top', color='red', marker='.', s=5)
  plt.xlabel('Right Ascension α [°]', fontsize = 10)
  plt.ylabel('Declination δ [°]', fontsize = 10)
  plt.title(f'Top Cluster of CWoLa Stars vs. GD-1, purity = {purity*100:.2f}%', fontsize = 12)
  plt.legend()
  plt.savefig(os.path.join(save_folder, "position.png"))
  plt.close()

  fig, axes = plt.subplots(1, 2, figsize=(10, 4), tight_layout=True)    
  axes[0].scatter(gd1_stars['rotpmra'], gd1_stars['rotpmdec'], label='GD-1 Stars', color='grey', marker='.', s=5)
  axes[0].scatter(top_cluster['rotpmra'], top_cluster['rotpmdec'], label='CWoLa Top', color='red', marker='.', s=5)
  axes[0].set_xlabel('Rotated Proper Motion (Right Ascension) μ_ϕcosλ [mas/yr]', fontsize = 10)
  axes[0].set_ylabel('Rotated Proper Motion (Declination) μ_λ [mas/yr]', fontsize = 10)
  axes[0].legend()

  axes[1].scatter(gd1_stars['b-r'], gd1_stars['g'], label='GD-1 Stars', color='grey', marker='.', s=5)
  axes[1].scatter(top_cluster['b-r'], top_cluster['g'], label='CWoLa Top', color='red', marker='.', s=5)
  axes[1].set_xlabel('Color b-r', fontsize = 10)
  axes[1].set_ylabel('Magnitude g', fontsize = 10)
  axes[1].legend()

  fig.savefig(os.path.join(save_folder, "pm_photometric.png"))
  plt.close()


  # top_stars top cwola star selections, gd1_stars all gd_1 stars
  # use euclidean distances to refine the number of non-stream stars in the top cluster for augmented stream labeling

  # normalize the input data
  if scan_var == 'rotpmdec':
    inputs = ['b-r', 'g', 'ra', 'dec', 'rotpmra']
    inputs_normalized = ['b-r_normalized', 'g_normalized', 'ra_normalized', 'dec_normalized', 'rotpmra_normalized']
  if scan_var == 'rotpmra':
    inputs = ['b-r', 'g', 'ra', 'dec', 'rotpmdec']
    inputs_normalized = ['b-r_normalized', 'g_normalized', 'ra_normalized', 'dec_normalized', 'rotpmdec_normalized']

  for input in inputs:
    top_cluster[f'{input}_normalized'] = (top_cluster[input] - top_cluster[input].mean()) / top_cluster[input].std()

  # split stream stars and non-labeled stars in all top stars
  true_stream = top_cluster[top_cluster['stream']==True]
  potential_stream = top_cluster[top_cluster['stream']==False]

  # calculate distances: distance_matrix
  all_distances = distance_matrix(true_stream[inputs_normalized].to_numpy(), potential_stream[inputs_normalized].to_numpy())
  closest_stream_star = true_stream.iloc[all_distances.argmin(axis=0)]
  deltas = potential_stream[inputs_normalized].to_numpy()-closest_stream_star[inputs_normalized].to_numpy()
  distances = np.sqrt(np.sum(deltas**2, axis=1))
  potential_stream['5d_distance'] = distances

  # separate top 10% closest stars to the stream stars as promising GD-1 candidates
  promising = potential_stream[(potential_stream['5d_distance'] < potential_stream['5d_distance'].quantile(0.1))]
  print(f"{len(promising)} promising GD-1 candidate stars found with pm {scan_var}.")
  promising = promising.sort_values('nn_score',ascending=False)
  promising.reset_index(inplace=True, drop=True)
  promising = promising[['patch', 'ra', 'ra_wrapped', 'dec', 'rotpmra', 'rotpmdec', 'b-r', 'g', '5d_distance', 'nn_score']]
  save_folder = f'../results/fullgd1/{scan_var}/promising'
  os.makedirs(save_folder, exist_ok=True)
  promising.to_hdf(os.path.join(save_folder, 'promising_stars.h5'), key='promising_stars', mode='w')

  # plot promising stars result over gd1 stars
  plt.figure()
  plt.scatter(gd1_stars['ra_wrapped'], gd1_stars['dec'], label='GD-1 Stars', color='grey', marker='.', s=5)
  scatter = plt.scatter(promising['ra_wrapped'], promising['dec'], label='Promising GD-1 Candidates', c=promising['nn_score'], cmap='Reds', marker='.', s=10, vmin=0.1, vmax=promising['nn_score'].max())
  plt.xlabel('Right Ascension α [°]', fontsize = 10)
  plt.ylabel('Declination δ [°]', fontsize = 10)
  plt.title(f'Promising GD-1 Candidates vs. GD-1', fontsize = 12)
  cbar = plt.colorbar(scatter)
  cbar.set_label('NN Score')
  plt.legend()
  plt.savefig(os.path.join(save_folder, "position.png")) 
  plt.close()

  fig, axes = plt.subplots(1, 2, figsize=(10, 4), tight_layout=True)    
  axes[0].scatter(gd1_stars['rotpmra'], gd1_stars['rotpmdec'], label='GD-1 Stars', color='grey', marker='.', s=5)
  scatter0 = axes[0].scatter(promising['rotpmra'], promising['rotpmdec'], label='Promising GD-1 Candidates', c=promising['nn_score'], cmap='Reds', marker='.', s=10, vmin=0.1, vmax=promising['nn_score'].max())
  axes[0].set_xlabel('Rotated Proper Motion (Right Ascension) μ_ϕcosλ [mas/yr]', fontsize = 10)
  axes[0].set_ylabel('Rotated Proper Motion (Declination) μ_λ [mas/yr]', fontsize = 10)
  cbar0 = fig.colorbar(scatter0, ax=axes[0])
  cbar0.set_label('NN Score')
  axes[0].legend()

  axes[1].scatter(gd1_stars['b-r'], gd1_stars['g'], label='GD-1 Stars', color='grey', marker='.', s=5)
  scatter1 = axes[1].scatter(promising['b-r'], promising['g'], label='Promising GD-1 Candidates', c=promising['nn_score'], cmap='Reds', marker='.', s=10, vmin=0.1, vmax=promising['nn_score'].max())
  axes[1].set_xlabel('Color b-r', fontsize = 10)
  axes[1].set_ylabel('Magnitude g', fontsize = 10)
  cbar1 = fig.colorbar(scatter1, ax=axes[1])
  cbar1.set_label('NN Score')
  axes[1].legend()

  fig.savefig(os.path.join(save_folder, "pm_photometric.png"))
  plt.close()











