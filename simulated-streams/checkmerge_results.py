### run once, single-process, after the torchrun training job finishes
### (torchrun blocks until every rank exits, so all partial files are
### guaranteed to be complete and present by the time this runs)

import glob
import os
import pandas as pd

from checkfunctions import get_results

save_folder = './results/checksim1'

partial_files = sorted(glob.glob(os.path.join(save_folder, 'df_test_rank*.h5')))
if not partial_files:
    raise FileNotFoundError(f'No df_test_rank*.h5 files found in {save_folder} — did the training job finish?')

print(f'Merging {len(partial_files)} partial result file(s): {partial_files}')
df_test_full = pd.concat([pd.read_hdf(f) for f in partial_files])
df_test_full.to_hdf(os.path.join(save_folder, 'df_test.h5'), key='data', mode='w')

## plot results
get_results(df_test_full, top_n=250, save_folder=os.path.join(save_folder, 'plots'))
