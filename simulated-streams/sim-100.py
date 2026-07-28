import sys
import pandas as pd
import torch

import sys
sys.path.append('../scripts')
from functions import *
 
## CHTC info: 1 GPU per job, $(Process) from the HTCondor job array (queue 100) is passed in as this stream's index.
stream_idx = int(sys.argv[1])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[stream {stream_idx}] using {device}')

df_all = pd.read_hdf('simulated-streams100.h5')
df = df_all[df_all['file'] == stream_idx]

df_regions = signal_sideband(df, bin_num=55, verbose=False)

save_folder = f'results/stream_{stream_idx}'
df_test = cwola_train(
    df_regions,
    pm_parameter='rotpmdec',
    dropout=0.2,
    k_folds=5,
    batch_size=10000,
    lr=0.001,
    patience=30,
    epochs=100,
    trainval_loops=3,
    save_folder=save_folder,
    wandbproj=None,   # wandb disabled
    device=device,
    rank=0,
    world_size=1,
)

purity = compute_purity(df_test, top_n=250)
print(f'[stream {stream_idx}] purity = {purity*100:.2f}%')
pd.DataFrame([{'stream': stream_idx, 'purity': purity, 'n_test': len(df_test)}]).to_csv(
    f'{save_folder}/purity.csv', index=False
)