
### imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import copy
import wandb
import os
from tqdm import tqdm



### loading .h5 file

def load_data(patch_idx):

    # load the .h5 data file into a dataframe, uses patch index (0-20)
    df = pd.read_hdf(f"../gaia-data/cleaned-data/gd1_patch{patch_idx}.h5")
    print(f"Data for Patch {patch_idx}")
    return df


def plot_data(df, save_folder = '../gaia-data/plots/patch'):

    if save_folder is not None:
        os.makedirs(save_folder, exist_ok=True)

    # create figure with 1 row and 3 columns for data subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), tight_layout=True)
    fig.suptitle('Full Patch')
    color = 'Greys'

    # plot position coordinates
    binspos = (np.linspace(-15,15,100), np.linspace(-15,15,100))
    h = axes[0].hist2d(df['rotra'], df['rotdec'], bins=binspos, cmap=color, cmin=0, vmax=250)
    axes[0].set_xlabel('Rotated Right Ascension ϕ [°]', fontsize = 10)
    axes[0].set_ylabel('Rotated Declination λ [°]', fontsize = 10)
    c = fig.colorbar(h[3], ax=axes[0])
    c.ax.set_title('Counts', fontsize=8)

    # plot proper motion coordinates
    binspm = (np.linspace(-20,20,100), np.linspace(-20,20,100))
    h = axes[1].hist2d(df['rotpmra'], df['rotpmdec'], bins=binspm, cmap=color, cmin=0)
    axes[1].set_xlabel('Rotated Proper Motion (Right Ascension) μ_ϕcosλ [mas/yr]', fontsize = 10)
    axes[1].set_ylabel('Rotated Proper Motion (Declination) μ_λ [mas/yr]', fontsize = 10)  
    c = fig.colorbar(h[3], ax=axes[1])
    c.ax.set_title('Counts', fontsize=8)

    # plot photometric figures
    binsfeat = (np.linspace(0,3,100), np.linspace(10,21,100))
    h = axes[2].hist2d(df['b-r'], df['g'], bins=binsfeat, cmap=color, cmin=0)
    axes[2].set_xlabel('Color b-r', fontsize = 10)
    axes[2].set_ylabel('Magnitude g', fontsize = 10)
    c = fig.colorbar(h[3], ax=axes[2])
    c.ax.set_title('Counts', fontsize=8)

    if save_folder is not None:
        fig.savefig(os.path.join(save_folder, "alldataplots.png"))

    df_stream = df[df['stream']==True]

    # create figure with 1 row and 3 columns for data subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), tight_layout=True)
    fig.suptitle('Labeled Stream Stars')
    color = 'Reds'

    # plot position coordinates
    binspos = (np.linspace(-15,15,100), np.linspace(-15,15,100))
    h = axes[0].hist2d(df_stream['rotra'], df_stream['rotdec'], bins=binspos, cmap=color, cmin=0)
    axes[0].set_xlabel('Rotated Right Ascension ϕ [°]', fontsize = 10)
    axes[0].set_ylabel('Rotated Declination λ [°]', fontsize = 10)
    c = fig.colorbar(h[3], ax=axes[0])
    c.ax.set_title('Counts', fontsize=8)

    # plot proper motion coordinates
    binspm = (np.linspace(-20,20,100), np.linspace(-20,20,100))
    h = axes[1].hist2d(df_stream['rotpmra'], df_stream['rotpmdec'], bins=binspm, cmap=color, cmin=0)
    axes[1].set_xlabel('Rotated Proper Motion (Right Ascension) μ_ϕcosλ [mas/yr]', fontsize = 10)
    axes[1].set_ylabel('Rotated Proper Motion (Declination) μ_λ [mas/yr]', fontsize = 10)  
    c = fig.colorbar(h[3], ax=axes[1])
    c.ax.set_title('Counts', fontsize=8)

    # plot photometric figures
    binsfeat = (np.linspace(0,3,100), np.linspace(10,21,100))
    h = axes[2].hist2d(df_stream['b-r'], df_stream['g'], bins=binsfeat, cmap=color, cmin=0)
    axes[2].set_xlabel('Color b-r', fontsize = 10)
    axes[2].set_ylabel('Magnitude g', fontsize = 10)
    c = fig.colorbar(h[3], ax=axes[2])
    c.ax.set_title('Counts', fontsize=8)

    if save_folder is not None:
        fig.savefig(os.path.join(save_folder, "streamdataplots.png"))



# define signal sideband region

def signal_sideband(df, save_folder = '../results/stream/patch', pm_parameter='rotpmdec', sig_factor=1, sb_factor=3, bin_num=55):
    if save_folder is not None:
        os.makedirs(save_folder, exist_ok=True)

    if pm_parameter == 'rotpmdec':
        pm_name = 'Rotated and Centered Proper Motion (DEC) μ_λ'
    if pm_parameter == 'rotpmra':
        pm_name = 'Rotated and Centered Proper Motion (RA) μ_ϕcosλ'

    print(f'Signal parameter: {pm_name}')
    print(f'Signal region factor = {sig_factor} and sideband region factor = {sb_factor}')

    df_stream = df[df['stream']==True]
    signal_parameter = df_stream[pm_parameter]
    pm_median = np.median(signal_parameter)
    pm_std = np.std(signal_parameter)

    print(f'Signal parameter median = {pm_median:.3f}')
    print(f'Signal parameter standard deviation = {pm_std:.3f}')

    sig_low = pm_median - sig_factor*pm_std
    sig_high = pm_median + sig_factor*pm_std
    sb_low = pm_median - sb_factor*pm_std
    sb_high = pm_median + sb_factor*pm_std
   
    print(f'Signal Region Range: [{sig_low:.3f}, {sig_high:.3f}]')
    print(f'Sideband Region Range: [{sb_low:.3f}, {sig_low:.3f}) and ({sig_high:.3f}, {sb_high:.3f}]')

    df_outer_region = df[(df[pm_parameter] < sb_low) | (df[pm_parameter] > sb_high)]
    df_outer_region_stream = df_outer_region[df_outer_region['stream']==True]
    df_outer_region_background = df_outer_region[df_outer_region['stream']==False]
    
    df_regions = df[(df[pm_parameter] >= sb_low) & (df[pm_parameter] <= sb_high)]
    # sideband region = 0, signal region = 1
    df_regions['region_label'] = np.where((df_regions[pm_parameter] >= sig_low) & (df_regions[pm_parameter] <= sig_high), 1, 0)
    
    df_sig_region = df_regions[df_regions['region_label']==1]
    df_sb_region = df_regions[df_regions['region_label']==0]
    
    df_sig_region_stream = df_sig_region[df_sig_region['stream']==True]
    df_sig_region_background = df_sig_region[df_sig_region['stream']==False]
    
    df_sb_region_stream = df_sb_region[df_sb_region['stream']==True]
    df_sb_region_background = df_sb_region[df_sb_region['stream']==False]
    
    print(f'Signal Region has {len(df_sig_region)} stars, {len(df_sig_region_stream)} stream and {len(df_sig_region_background)} background.')
    print(f'Sideband Region has {len(df_sb_region)} stars, {len(df_sb_region_stream)} stream and {len(df_sb_region_background)} background.')
    print(f'Outer Region has {len(df_outer_region)} stars, {len(df_outer_region_stream)} stream and {len(df_outer_region_background)} background.')

    bins = np.linspace(sb_low - (sig_low - sb_low), sb_high + (sb_high - sig_high), bin_num)
    plt.hist(df_sig_region_stream[pm_parameter], label='Signal Region', color='red', alpha=1, bins=bins)
    plt.hist(df_sb_region_stream[pm_parameter], label='Sideband Region', color='red', alpha=0.5, bins=bins)
    plt.hist(df_outer_region_stream[pm_parameter], label='Outer Region', color='red', alpha=0.25, bins=bins)
    plt.xlabel(f'{pm_name} [mas/yr]')
    plt.ylabel('Number of Stars')
    plt.title('Stream Stars in Patch')
    plt.legend()
    plt.show()
    plt.close()
    if save_folder is not None:
        plt.savefig(os.path.join(save_folder, "pmstreamdistribution.png"))
    
    plt.hist(df_sig_region_background[pm_parameter], label='Signal Region', color='grey', alpha=1, bins=bins)
    plt.hist(df_sb_region_background[pm_parameter], label='Sideband Region', color='grey', alpha=0.5, bins=bins)
    plt.hist(df_outer_region_background[pm_parameter], label='Outer Region', color='grey', alpha=0.25, bins=bins)
    plt.xlabel(f'{pm_name} [mas/yr]')
    plt.ylabel('Number of Stars')
    plt.title('Background Stars in Patch')
    plt.legend()
    plt.show()
    plt.close()
    if save_folder is not None:
        plt.savefig(os.path.join(save_folder, "pmbackgrounddistribution.png"))
            
    return df_regions



def cwola_train(df, pm_parameter='rotpmdec', dropout=0.2, k_folds=5, batch_size=10000, lr=0.001, patience=30, epochs=100, trainval_loops=3, save_folder='../results/stream/patch', wandbproj='sim-1patch'):

    use_wandb = wandbproj is not None

    os.makedirs(save_folder, exist_ok=True)

    # training variables
    if pm_parameter == 'rotpmdec':
        training_vars = ['b-r', 'g', 'rotra', 'rotdec', 'rotpmra']
        print('Proper Motion Parameter: Rotated and Centered Proper Motion (DEC) μ_λ')
    if pm_parameter == 'rotpmra':
        training_vars = ['b-r', 'g', 'rotra', 'rotdec', 'rotpmdec']
        print('Proper Motion Parameter: Rotated and Centered Proper Motion (RA) μ_ϕcosλ')

    # model architecture
    class NeuralNetwork(nn.Module):
        def __init__(self, input_dim = 5, hidden_dim = 256, output_dim = 1, dropout=dropout): # create tool, parameters for recalling, call function is used after
            super().__init__() # access all methods, attributes, etc of class
            self.NeuralNetwork = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout),       
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout),       
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout),       
                nn.Linear(hidden_dim, output_dim),
                nn.Sigmoid()
            )
        def forward(self, x):
            return self.NeuralNetwork(x)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # k-folding
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=18)

    fold_stars = [] # will be a list of k (5) arrays, representing the k folds
    for fold_idx, (train_index, test_index) in enumerate(skf.split(df[training_vars], df['region_label'])):
        fold_stars.append(test_index)
    fold_labels = np.arange(len(fold_stars)) # fold labels is an array of values 0-4, representing indices/labels of the 5 folds

    test_dfs = []
    for test_idx in my_test_folds:
        print(f'Test fold {test_idx}...')
        
        test_stars = fold_stars[test_idx]
        save_folder_test = os.path.join(save_folder, "test_fold_{}".format(test_idx))
        os.makedirs(save_folder_test, exist_ok=True)

        # cycles through remaining sets (minus the test set) for validation sets
        for val_idx in np.delete(fold_labels, test_idx):
            print(f'Validation fold {val_idx}...')
            val_stars = fold_stars[val_idx]
    
            train_indices = np.delete(fold_labels, [test_idx, val_idx])
            train_stars = np.concatenate([fold_stars[train_idx] for train_idx in train_indices])
    
            # create data frames with indices
            df_train = df.iloc[train_stars]
            df_val = df.iloc[val_stars]
            df_test = df.iloc[test_stars]
    
            train_x = df_train[training_vars]
            train_y = df_train['region_label'].to_numpy()
    
            val_x = df_val[training_vars]
            val_y = df_val['region_label'].to_numpy()
    
            test_x = df_test[training_vars]
            test_y = df_test['region_label'].to_numpy()
            test_y_true = df_test['stream'].to_numpy(dtype=int)

            # scale data
            scaler = StandardScaler()
            
            train_x = scaler.fit_transform(train_x) # fit_transform called only on train data
            val_x = scaler.transform(val_x)
            test_x = scaler.transform(test_x)
                    
            # create tensors, datasets, and loaders
            train_x_tensor = torch.tensor(np.array(train_x), dtype=torch.float32, device=device)
            train_y_tensor = torch.tensor(train_y, dtype=torch.float32, device=device)
                
            val_x_tensor = torch.tensor(np.array(val_x), dtype=torch.float32, device=device)
            val_y_tensor = torch.tensor(val_y, dtype=torch.float32, device=device)
                
            test_x_tensor = torch.tensor(np.array(test_x), dtype=torch.float32, device=device)
            test_y_tensor = torch.tensor(test_y, dtype=torch.float32, device=device)
            test_y_true_tensor = torch.tensor(test_y_true, dtype=torch.long, device=device)
                
            train_dataset = TensorDataset(train_x_tensor, train_y_tensor)
            val_dataset = TensorDataset(val_x_tensor, val_y_tensor)
            test_dataset = TensorDataset(test_x_tensor, test_y_tensor, test_y_true_tensor)
                
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            best_loop_val_loss = float('inf')

            test_nn_scores = []
            
            for n in range(trainval_loops):
                print(f'Starting training loop {n}')

                # initialize model
                if use_wandb:
                    run = wandb.init(project=wandbproj,
                                group=f"fold_{test_idx}",
                                job_type=f"val_set_{val_idx}",
                                name=f"test_{test_idx}_val_{val_idx}_train_{n}",
                                config={"test_fold": test_idx, "val_set": val_idx, "train_loop": n},
                                reinit=True)
                model = NeuralNetwork(input_dim = 5, hidden_dim = 256, output_dim = 1, dropout=dropout).to(device)
                model = torch.compile(model)
                loss_fn = nn.BCELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
                best_val_loss = float('inf')
                patience_counter = 0
                
                for epoch in tqdm(range(num_epochs), desc="Epoch {epoch}"):
                
                    # training
                    model.train()
                    total_train_loss = torch.tensor(0.0, device=device)
                    train_correct = torch.tensor(0.0, device=device)
                    counts = 0
                    
                    for inputs, region_labels in train_loader:
                        optimizer.zero_grad()
                        out = model(inputs).squeeze(1)
                        loss = loss_fn(out, region_labels)
                        loss.backward()
                        optimizer.step()
                        total_train_loss += loss.detach() * inputs.size(0)
                    
                        predictions = (out > 0.5).float()
                        train_correct += (predictions == region_labels).sum()
                        counts += region_labels.size(0)
                    
                    avg_train_loss = (total_train_loss / counts).item()  
                    train_accuracy = (train_correct / counts).item()
                    
                    # validation
                    model.eval()
                    total_val_loss = torch.tensor(0.0, device=device)
                    val_correct = torch.tensor(0.0, device=device)
                    counts = 0
                    with torch.no_grad():
                        for inputs, region_labels in val_loader:
                            out = model(inputs).squeeze(1)  # forward pass
                            loss = loss_fn(out, region_labels)  # compute loss
                            total_val_loss += loss.detach() * inputs.size(0)
                    
                            predictions = (out > 0.5).float()
                            val_correct += (predictions == region_labels).sum()
                            counts += region_labels.size(0)
                
                    avg_val_loss = (total_val_loss / counts).item()
                    val_accuracy = (val_correct / counts).item()

                    if use_wandb:
                        wandb.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss,
                                 "train_acc": train_accuracy, "val_acc": val_accuracy, "epoch": epoch})

                    # early stopping
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        patience_counter = 0
                        best_model = copy.deepcopy(model.state_dict())
                    else:
                        patience_counter += 1
                
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch}")
                        break
                
                if best_val_loss < best_loop_val_loss:
                    best_loop_val_loss = best_val_loss
                    lowest_val_loss_model = best_model ## model parameters from epoch with the lowest validation loss

                if use_wandb:
                    run.finish()
                    
            print(f'Best validation loss for validation fold label {val_idx} = {best_val_loss}')
            model_save_path = os.path.join(save_folder_test, "val_set_{}_best_model".format(val_idx))
            torch.save(lowest_val_loss_model, model_save_path) ## model parameters from training run with the lowest validation loss

            model.load_state_dict(lowest_val_loss_model)

            model.eval()
            test_loss, test_correct, counts = 0.0, 0.0, 0
            raw_scores = []
            
            with torch.no_grad():
                for inputs, region_labels, true_labels in test_loader:
                    out = model(inputs).squeeze()
                    raw_scores.extend(out.squeeze().cpu().numpy())
                    loss = loss_fn(out, region_labels)
                    test_loss += loss.item() * inputs.size(0)
            
                    scores = (out > 0.5).float()
                    test_correct += (scores == region_labels).sum().item()
                    counts += region_labels.size(0)
            
            test_nn_scores.append(raw_scores)
            print(f'Test scores saved for test fold {test_idx}.')

        print('Averaging nn_scores for each star from the 4 best models...')
        df_test['nn_score'] = np.mean(test_nn_scores, axis=0)
        test_dfs.append(df_test)

    df_test_full = pd.concat([df for df in test_dfs])
    out_path = f'{save_folder}/df_test.h5'
    df_test_full.to_hdf(out_path, key='data', mode='w')
    print('Full test df with all stars saved!')

    return df_test_full


def fiducial_cuts(df): # CWOLA Stellar Stream fiducial cuts https://arxiv.org/pdf/2305.03761
    df = df[df['g'] < 20.2] # ensures uniform acceptance by Gaia satellite
    df = df[(np.abs(df['rotpmdec']) > 2) | (np.abs(df['rotpmra']) > 2)] # remove distant stars concentrated near zero proper motion, not equally distributed throughout patch
    df = df[(df['b-r'] >= 0.5) & (df['b-r'] <=1 )] # isolates old and low-metallicity stellar streams in color space
    return df


def compute_purity_and_completeness(df, top_n=250):
    df_ranked = df.sort_values(by='nn_score', ascending=False)
    df_top = df_ranked[:top_n]
    purity = len(df_top[df_top['stream']==True]) / len(df_top)
    completeness = len(df_top[df_top['stream']==True])/len(df[df['stream']==True])
    return purity, completeness


def get_results(df, top_n=250, fid_cuts=True, save_folder='../results/streams/patch'):

    if fid_cuts:
        df = fiducial_cuts(df)
        os.makedirs(os.path.join(save_folder, 'fid_cuts'), exist_ok=True)
        print('Plotting after fiducial cuts...')
    else:
        os.makedirs(os.path.join(save_folder, 'no_fid_cuts'), exist_ok=True)
        print('Plotting before fiducial cuts..')
    
    df_signalreg = df[df['region_label']==1]
    df_backgroundreg = df[df['region_label']==0]

    df_streamstar = df[df['stream']==True]
    df_backgroundstar = df[df['stream']==False]

    # NN scores
    plt.hist(df_signalreg['nn_score'], label='Signal Region', histtype='step', bins=50)
    plt.hist(df_backgroundreg['nn_score'], label='Background Region', histtype='step', bins=50)
    plt.xlabel('NN Score')
    plt.ylabel('Number of Stars')
    plt.legend()
    plt.savefig(os.path.join(save_folder, "nnscoreregions.png"))
    plt.close()

    plt.hist(df_streamstar['nn_score'], label='Stream Stars', color='red', histtype='step', bins=50)
    plt.hist(df_backgroundstar['nn_score'], label='Background Stars', color='grey', histtype='step', bins=50)
    plt.yscale('log')
    plt.xlabel('NN Score')
    plt.ylabel('Number of Stars')
    plt.legend()
    plt.savefig(os.path.join(save_folder, "nnscorestars.png"))
    plt.close()

    # top N stars
    df_ranked = df.sort_values(by='nn_score', ascending=False)
    df_top = df_ranked[:top_n]
    outpath = f'{save_folder}/df_top.h5'
    df_top.to_hdf(outpath, key='df_top', mode='w')

    # purity and completeness
    purity, completeness = compute_purity_and_completeness(df, top_n=top_n)
    print(f'Top {top_n} ranked stars: Purity = {purity*100:.2f}%')
    print(f'Top {top_n} ranked stars: Completeness = {completeness*100:.2f}%')

    # plot cwola stars on top of gaia stars
    df_top_signal = df_top[df_top['stream']==True]
    df_top_background = df_top[df_top['stream']==False]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), tight_layout=True)
    fig.suptitle(f'Patch 0: CWoLa Top Matches, purity = {purity*100:.2f}%')
    
    axes[0].scatter(df_streamstar['ra'], df_streamstar['dec'], label='GD-1 Stars', color='grey', marker='.', s=5)
    axes[0].scatter(df_top_signal['ra'], df_top_signal['dec'], label='CWoLa Matches', color='red', marker='.', s=5)
    axes[0].scatter(df_top_background['ra'], df_top_background['dec'], label='CWoLa Non-Matches', color='blue', marker='.', s=5)
    axes[0].set_xlabel('Right Ascension α [°]', fontsize = 10)
    axes[0].set_ylabel('Declination δ [°]', fontsize = 10)
    axes[0].legend()

    axes[1].scatter(df_streamstar['rotpmra'], df_streamstar['rotpmdec'], label='GD-1 Stars', color='grey', marker='.', s=5)
    axes[1].scatter(df_top_signal['rotpmra'], df_top_signal['rotpmdec'], label='CWoLa Matches', color='red', marker='.', s=5)
    axes[1].scatter(df_top_background['rotpmra'], df_top_background['rotpmdec'], label='CWoLa Non-Matches', color='blue', marker='.', s=5)
    axes[1].set_xlabel('Rotated Proper Motion (Right Ascension) μ_ϕcosλ [mas/yr]', fontsize = 10)
    axes[1].set_ylabel('Rotated Proper Motion (Declination) μ_λ [mas/yr]', fontsize = 10)

    axes[2].scatter(df_streamstar['b-r'], df_streamstar['g'], label='GD-1 Stars', color='grey', marker='.', s=5)
    axes[2].scatter(df_top_signal['b-r'], df_top_signal['g'], label='CWoLa Matches', color='red', marker='.', s=5)
    axes[2].scatter(df_top_background['b-r'], df_top_background['g'], label='CWoLa Non-Matches', color='blue', marker='.', s=5)
    axes[2].set_xlabel('Color b-r', fontsize = 10)
    axes[2].set_ylabel('Magnitude g', fontsize = 10)

    plt.tight_layout()
    fig.savefig(os.path.join(save_folder, "finalstars.png"))
    plt.close()

    



### cleaning up and converting raw .npy data to .h5, functions from Dr. Pettee's repo https://github.com/hep-lbdl/GaiaCWoLa/tree/main

def angular_distance(angle1,angle2): # via_machinae function https://arxiv.org/abs/2104.12789
    # inputs are np arrays of [ra,dec]
    deltara=np.minimum(np.minimum(np.abs(angle1[:,0]-angle2[:,0]+360),
                                  np.abs(angle1[:,0]-angle2[:,0])),
                                  np.abs(angle1[:,0]-angle2[:,0]-360))
    deltadec=np.abs(angle1[:,1]-angle2[:,1])
    return np.sqrt(deltara**2+deltadec**2)
# function that cross checks GD-1 stars with raw patch data, returns boolean array

def FilterGD1(stars, gd1_stars):
    gd1stars=np.zeros(len(stars))
    for x in tqdm(gd1_stars):
        ra=x['ra']
        dec=x['dec']
        pmra=x['pmra']
        pmdec=x['pmdec']
        foundlist=angular_distance(np.dstack((stars[:,3],stars[:,2]))[0],np.array([[ra,dec]]))
        foundlist=np.sqrt(foundlist**2+(stars[:,0]-pmdec)**2+(stars[:,1]-pmra)**2)   
        foundlist=foundlist<.0001
        if len(np.argwhere(foundlist))>1:
            print(foundlist)
        if len(np.argwhere(foundlist))==1:
            gd1stars+=foundlist
    gd1stars=gd1stars.astype('bool')
    return gd1stars,stars[gd1stars]
