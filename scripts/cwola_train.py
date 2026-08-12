import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import copy
import wandb
import os
from tqdm.auto import tqdm


def cwola_train(df, pm_parameter='rotpmdec', dropout=0.2, k_folds=5, batch_size=10000, lr=0.001, patience=30, epochs=100, trainval_loops=3, save_folder='../results/patch/training', wandbproj=None):

    use_wandb = wandbproj is not None

    os.makedirs(save_folder, exist_ok=True)

    # denotes training variables, separate from proper motion parameter used to define signal/sideband regions
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

    fold_stars = [] # will become a list of k (5) arrays, representing the k folds
    for fold_idx, (train_index, test_index) in enumerate(skf.split(df[training_vars], df['region_label'])):
        fold_stars.append(test_index)
    fold_labels = np.arange(len(fold_stars)) # fold labels is an array of values 0-4, representing indices/labels of the 5 folds

    test_dfs = []
    for test_idx in fold_labels: # loops through each fold as the test fold
        print('-----------------------------------------------')
        print(' ')
        print(f'Test fold {test_idx}...')
        
        test_stars = fold_stars[test_idx]
        save_folder_test = os.path.join(save_folder, "test_fold_{}".format(test_idx))
        os.makedirs(save_folder_test, exist_ok=True)

        test_nn_scores = []
        for val_idx in np.delete(fold_labels, test_idx): # cycles through remaining sets (minus the test set) for validation sets
            print(f'Validation fold {val_idx}...')
            val_stars = fold_stars[val_idx]
    
            train_indices = np.delete(fold_labels, [test_idx, val_idx])
            train_stars = np.concatenate([fold_stars[train_idx] for train_idx in train_indices])
    
            df_train = df.iloc[train_stars] # create data frames with indices
            df_val = df.iloc[val_stars]
            df_test = df.iloc[test_stars]
    
            train_x = df_train[training_vars]
            train_y = df_train['region_label'].to_numpy()
    
            val_x = df_val[training_vars]
            val_y = df_val['region_label'].to_numpy()
    
            test_x = df_test[training_vars]
            test_y = df_test['region_label'].to_numpy()
            test_y_true = df_test['stream'].to_numpy(dtype=int)

            scaler = StandardScaler() # scale data to standardize with mean of 0 and standard deviation of 1
            
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
            
            for n in range(trainval_loops): # trains the model trainval_loops number of times (equal to the number of remaining folds, used as training)
                print(f'Starting training loop {n+1}')

                if use_wandb:
                    run = wandb.init(project=wandbproj,
                                group=f"fold_{test_idx}",
                                job_type=f"val_set_{val_idx}",
                                name=f"test_{test_idx}_val_{val_idx}_train_{n}",
                                config={"test_fold": test_idx, "val_set": val_idx, "train_loop": n},
                                reinit=True)
                    
                # initialize model upon each training loop
                model = NeuralNetwork(input_dim = 5, hidden_dim = 256, output_dim = 1, dropout=dropout).to(device)
                loss_fn = nn.BCELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
                best_val_loss = float('inf')
                patience_counter = 0
                
                for epoch in tqdm(range(epochs), desc='Epochs'):
                
                    # training
                    model.train()
                    total_train_loss = torch.tensor(0.0, device=device)
                    train_correct = torch.tensor(0.0, device=device)
                    counts = 0
                    
                    for inputs, region_labels in train_loader:
                        inputs = inputs.to(device)
                        region_labels = region_labels.to(device)

                        optimizer.zero_grad()
                        out = model(inputs).squeeze(1) # forward pass
                        loss = loss_fn(out, region_labels) # compute loss
                        loss.backward()
                        optimizer.step()
                        total_train_loss += loss.detach() * inputs.size(0)
                    
                        predictions = (out > 0.5).float()
                        train_correct += (predictions == region_labels).sum()
                        counts += region_labels.size(0)
                    
                    avg_train_loss = (total_train_loss / counts).item()  
                    train_accuracy = (train_correct / counts).item()
                    
                    # validation, used to track loss/accuracy, monitor for early stopping
                    model.eval()
                    total_val_loss = torch.tensor(0.0, device=device)
                    val_correct = torch.tensor(0.0, device=device)
                    counts = 0
                    with torch.no_grad():
                        for inputs, region_labels in val_loader:
                            inputs = inputs.to(device)
                            region_labels = region_labels.to(device)
                            
                            out = model(inputs).squeeze(1)
                            loss = loss_fn(out, region_labels)
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
                        print(f"Early stopping at epoch {epoch} with average val_loss {avg_val_loss}")
                        break
                
                if best_val_loss < best_loop_val_loss:
                    best_loop_val_loss = best_val_loss
                    lowest_val_loss_model = best_model ## model parameters from epoch with the lowest validation loss

                if use_wandb:
                    run.finish()
                    
            print(f'Best validation loss for validation fold label {val_idx} = {best_val_loss}')
            model_save_path = os.path.join(save_folder_test, "val_set_{}_best_model".format(val_idx))
            torch.save(lowest_val_loss_model, model_save_path) ## model parameters from training run with the lowest validation loss

            model.load_state_dict(lowest_val_loss_model) ## test set evaluated on the best model from each of the 4 validation sets

            model.eval()
            test_loss, test_correct, counts = 0.0, 0.0, 0
            raw_scores = []
            
            with torch.no_grad():
                for inputs, region_labels, true_labels in test_loader:
                    inputs = inputs.to(device)
                    region_labels = region_labels.to(device)
                    out = model(inputs).squeeze()
                    raw_scores.extend(out.squeeze().cpu().numpy())
            
            test_nn_scores.append(raw_scores)
            print(f'Test scores saved for test fold {test_idx} val fold {val_idx}.')

        print('Averaging nn_scores for each star from the 4 best models...')
        df_test['nn_score'] = np.mean(test_nn_scores, axis=0)
        test_dfs.append(df_test)

    df_test_full = pd.concat([df for df in test_dfs])
    out_path = f'{save_folder}/df_test.h5'
    df_test_full.to_hdf(out_path, key='data', mode='w')
    print('Full test df with all stars saved!')

    return df_test_full