from functions import *
import pandas as pd
from scipy.spatial import distance_matrix

## reproducibility with rotpmdec and comparison with rotpmra

scan_vars = ['rotpmdec', 'rotpmra']

for i in range(len(scan_vars)):
  scan_var = scan_vars[i]

  top_dfs = []
  gd1stars_dfs = []
  for idx in tqdm(range(21), desc='Patch'):
  
    df_test = pd.read_hdf(f'..results/patch_{idx}/{scan_var}/training/df_test.h5')
    df_gd1stars = df_test[df_test['stream']==True]
    gd1stars_dfs.append(df_gd1stars)
    
    df_top = pd.read_hdf(f'../results/patch_{idx}/{scan_var}/df_top.h5')
    top_dfs.append(df_top)
  
    print(f'Successfully loaded df_gd1 and df_top for Patch {idx} and scan variable {scan_var}')
  
  top_stars = pd.concat([df for df in top_dfs])
  gd1_stars = pd.concat([df for df in gd1stars_dfs])

  top_stars.to_hdf(f'../results/fullgd1/{scan_var}/raw/top_stars.h5', key='top_stars', mode='w')

  # to check if neural network scores are the same or different between duplicates
  if scan_var == 'rotpmdec':
    check_cols = ['b-r', 'g', 'ra', 'dec', 'pmra']
  if scan_var == 'rotpmra':
    check_cols = ['b-r', 'g', 'ra', 'dec', 'pmdec']
  duplicate_rows = top_stars[top_stars.duplicated(subset=check_cols, keep=False)].sort_values(by=check_cols)
  print(f'Duplicate rows found: {len(duplicate_rows)}')
  print(duplicate_rows)
  
  top_stars = top_stars.drop_duplicates(subset=check_cols,ignore_index=True)
  top_stream_stars = top_stars[top_stars['stream']==True]
  top_background_stars = top_stars[top_stars['stream']==False]
  gd1_stars = gd1_stars.drop_duplicates(subset=check_cols,ignore_index=True)

  print(f'Total number of unique CWoLa top stars = {len(top_stars)}')
  print(f'Total number of true stream stars of the unique CWoLa top stars = {len(top_stream_stars)}/{len(top_stars)} ({len(top_stream_stars)/len(top_stars)*100:.2f}%)')

  save_folder = f'../results/fullgd1/{scan_var}/raw'
  
  plt.figure()
  plt.scatter(gd1_stars['ra'], gd1_stars['dec'], label='GD-1 Stars', color='grey', marker='.', s=5)
  plt.scatter(top_stream_stars['ra'], top_stream_stars['dec'], label='CWoLa Matches', color='red', marker='.', s=5)
  plt.scatter(top_background_stars['ra'], top_background_stars['dec'], label='CWoLa Non-Matches', color='blue', marker='.', s=5)
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

  axes[1].scatter(gd1_stars['b-r'], gd1_stars['g'], label='GD-1 Stars', color='grey', marker='.', s=5)
  axes[1].scatter(top_stream_stars['b-r'], top_stream_stars['g'], label='CWoLa Matches', color='red', marker='.', s=5)
  axes[1].scatter(top_background_stars['b-r'], top_background_stars['g'], label='CWoLa Non-Matches', color='blue', marker='.', s=5)
  axes[1].set_xlabel('Color b-r', fontsize = 10)
  axes[1].set_ylabel('Magnitude g', fontsize = 10)

  fig.savefig(os.path.join(save_folder, "pm_photometric.png"))
  plt.close()

  save_folder = f'../results/fullgd1/{scan_var}/k_means'

  ## kmeans here !!!!!
  kmeans = KMeans(n_clusters=2, random_state=42)
  kmeans.fit(top_stars[['rotpmra', 'rotpmdec']])
  top_stars['kmeans_label'] = kmeans.labels_
  group_0 = top_stars[top_stars['kmeans_label'] == 0]
  print(f'Group 0: {len(group_0)} stars')
  group_1 = top_stars[top_stars['kmeans_label'] == 1]
  print(f'Group 1: {len(group_1)} stars')
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
      print(f'Top cluster is Group 0 with {len(group_0)} stars')
  else:
      top_cluster = group_1
      print(f'Top cluster is Group 1 with {len(group_1)} stars')

  top_cluster_stream = top_cluster[top_cluster['stream']==True]
  top_cluster_background = top_cluster[top_cluster['stream']==False]

  # plot top cluster result over gd1 stars

  plt.figure()
  plt.scatter(gd1_stars['ra'], gd1_stars['dec'], label='GD-1 Stars', color='grey', marker='.', s=5)
  plt.scatter(top_cluster_stream['ra'], top_cluster_stream['dec'], label='CWoLa Matches', color='red', marker='.', s=5)
  plt.scatter(top_cluster_background['ra'], top_cluster_background['dec'], label='CWoLa Non-Matches', color='blue', marker='.', s=5)
  plt.xlabel('Right Ascension α [°]', fontsize = 10)
  plt.ylabel('Declination δ [°]', fontsize = 10)
  plt.legend()
  plt.savefig(os.path.join(save_folder, "position.png"))
  plt.close()
  
  fig, axes = plt.subplots(1, 2, figsize=(10, 4), tight_layout=True)    
  axes[0].scatter(gd1_stars['rotpmra'], gd1_stars['rotpmdec'], label='GD-1 Stars', color='grey', marker='.', s=5)
  axes[0].scatter(top_cluster_stream['rotpmra'], top_cluster_stream['rotpmdec'], label='CWoLa Matches', color='red', marker='.', s=5)
  axes[0].scatter(top_cluster_background['rotpmra'], top_cluster_background['rotpmdec'], label='CWoLa Non-Matches', color='blue', marker='.', s=5)
  axes[0].set_xlabel('Rotated Proper Motion (Right Ascension) μ_ϕcosλ [mas/yr]', fontsize = 10)
  axes[0].set_ylabel('Rotated Proper Motion (Declination) μ_λ [mas/yr]', fontsize = 10)

  axes[1].scatter(gd1_stars['b-r'], gd1_stars['g'], label='GD-1 Stars', color='grey', marker='.', s=5)
  axes[1].scatter(top_cluster_stream['b-r'], top_cluster_stream['g'], label='CWoLa Matches', color='red', marker='.', s=5)
  axes[1].scatter(top_cluster_background['b-r'], top_cluster_background['g'], label='CWoLa Non-Matches', color='blue', marker='.', s=5)
  axes[1].set_xlabel('Color b-r', fontsize = 10)
  axes[1].set_ylabel('Magnitude g', fontsize = 10)

  fig.savefig(os.path.join(save_folder, "pm_photometric.png"))
  plt.close()

  purity = len(top_cluster_stream) / len(top_cluster)
  print(f'Purity of the top cluster: {purity*100:.2f}% ({len(top_cluster_stream)}/{len(top_cluster)})')
  completeness = len(top_cluster_stream) / len(gd1_stars)
  print(f'Completeness of the top cluster vs GD-1: {completeness*100:.2f}% ({len(top_cluster_stream)}/{len(gd1_stars)})')


  # top_stars top cwola star selections, gd1_stars all gd_1 stars
  # use euclidean distances to refine the number of non-stream stars in the top cluster for augmented stream labeling

  # normalize the input data
  if scan_var == 'rotpmdec':
      inputs = ['b-r', 'g', 'rotra', 'rotdec', 'rotpmra']
      inputs_normalized = ['b-r_normalized', 'g_normalized', 'rotra_normalized', 'rotdec_normalized', 'rotpmra_normalized']
  if scan_var == 'rotpmra':
      inputs = ['b-r', 'g', 'rotra', 'rotdec', 'rotpmdec']
      inputs_normalized = ['b-r_normalized', 'g_normalized', 'rotra_normalized', 'rotdec_normalized', 'rotpmdec_normalized']

  for input in inputs:
      top_stars[f'{input}_normalized'] = (top_stars[input] - top_stars[input].mean()) / top_stars[input].std()

  # split stream stars and non-labeled stars in all top stars
  true_stream = top_stars[top_stars['stream']==True]
  potential_stream = top_stars[top_stars['stream']==False]

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
  promising.to_hdf(f'../results/fullgd1/{scan_var}/promising_stars.h5', key='promising_stars', mode='w')










