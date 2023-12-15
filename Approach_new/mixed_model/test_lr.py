# %%
from generate_mdps import generate_datsets
from dataset import MDPDataset, AllNodeFeatures, InMemoryMDPDataset, TransitionsOnEdge
from experiment import Experiment
from MDP_helpers import calculate_gap, multiclass_recall_score

# %%
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import GCN, GAT
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import random_split
from collections import defaultdict
from sklearn.metrics import recall_score
from sklearn.model_selection import KFold
from torch.utils.data import SubsetRandomSampler

from time import time
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

torch.cuda.manual_seed(12345)
np.random.seed(12345)

####################################################################################################
# Configs
experiment_name = "GAT_NoLR_test"
hparam_file = "hparams_1000_loss"
ascending = True
edge_attributes = True
gnn_model = GAT
pre_transform = TransitionsOnEdge()
lr_shedule = False
step_lr_step_size=100
learning_rates = [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
early_stopping = True
early_stopping_patience = 1000

training_configs = {
    "N_epochs": 10000,
    "N_mdps": 100,
    "Re-create data": True,
    "MDP sizes": [
        # {"N_sites":4, "N_species":20, "K":6, "device": None},
        {"N_sites":5, "N_species":20, "K":7, "device": None},
        # {"N_sites":6, "N_species":20, "K":8, "device": None},
        # {"N_sites":7, "N_species":20, "K":9, "device": "cpu"},
    ],
    "train proportion": 0.8,
    "N_trials": 5
}
###################################################################################################

# Create experiment folder
experiment_exists = os.path.isfile(f"Results/{experiment_name}/{hparam_file}")
if not experiment_exists:
    raise Exception("Hparams do not yet exist!")

# Get device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using Device: {device}")

experiment = Experiment(savefile=f"Results/{experiment_name}/{hparam_file}")

print("Top 10 Hparams (including previous runs)")
hparams = pd.DataFrame(experiment.load()).sort_values(by="score", ascending=ascending).head()
print(hparams)

best_params = hparams.iloc[0].to_dict()

###################################################################################################
print("\n")
print("Starting training loops")
N_datasets = training_configs["N_mdps"]
N_epochs = training_configs["N_epochs"]

for mdp_configs in training_configs["MDP sizes"]:
    print("\n\n")
    N_sites = mdp_configs["N_sites"]
    N_species = mdp_configs["N_species"]
    K = mdp_configs["K"]
    N_states = 3**N_sites

    if mdp_configs["device"]:
        device = mdp_configs["device"]

    print(f"MDP Data: N_states: {N_states}")
    dataset_folder = f"Reserve_MDP_{N_states}_{K}"
    generate_datsets(N_sites, N_species, K, N_datasets, remove_previous=training_configs["Re-create data"], folder=dataset_folder)

    print("Loading data into dataloader")
    print(pre_transform)
    dataset = MDPDataset(f"datasets/{dataset_folder}", pre_transform=pre_transform)
    print(dataset[0])
    if torch.all(dataset[0].R == dataset[5].R):
        raise Exception("Datasets are likely identical!!")
    

    for lr in learning_rates:
        filename = f'Reserve_MDP_{N_states}_{K}_lr_{lr*10:.4f}'.replace(".", "")
        folder_exists = os.path.isdir(f"Results/{experiment_name}/{filename}")
        if folder_exists:
            raise Exception("Folder exists: Experiment Exists Already!")
        else:
            print("Creating experiment folder: Results/", experiment_name, "/", filename)
            os.mkdir(f"Results/{experiment_name}/{filename}")

        all_results = defaultdict(lambda : defaultdict(list))
        print(f"Starting K-fold cross validation using device {device}")
        kfold = KFold(n_splits=training_configs["N_trials"], shuffle=True)

        min_epochs = N_epochs
        for trial_num, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
            train_sampler = SubsetRandomSampler(train_ids)
            test_sampler = SubsetRandomSampler(test_ids)

            train_data = DataLoader(dataset, batch_size=1, sampler=train_sampler)
            test_data = DataLoader(dataset, batch_size=1, sampler=test_sampler)

            trial_name = f"trial_{trial_num}"

            hidden_channels = int(best_params['hidden_channels'])
            num_layers = 1
            dropout = best_params['dropout']
            # lr = best_params['lr']
            weight_decay = best_params['weight_decay']
            if lr_shedule:
                gamma = best_params['gamma']

            model = gnn_model(
                in_channels=dataset[0].x.shape[1], 
                out_channels=K, 
                hidden_channels=hidden_channels, 
                num_layers=num_layers, 
                dropout=dropout
            ).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            if lr_shedule:
                lr_sheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_lr_step_size, gamma=gamma)

            epochs = tqdm(range(N_epochs))
            early_stopping_counter = 0
            best_loss = float('inf')
            for epoch in epochs:
                model.train()
                optimizer.zero_grad()     

                loss = 0
                for data in train_data:
                    if edge_attributes:
                        pred = model(
                            x = data.x.to(device), 
                            edge_index=data.edges.to(device), 
                            edge_attr=data.edge_features.to(device)
                        )
                    else:
                        pred = model(
                            x = data.x.to(device), 
                            edge_index=data.edges.to(device), 
                        )  
                    loss += F.cross_entropy(pred, data.k_labels.to(device))

                loss /= len(train_data) 
                loss.backward()
                optimizer.step()
                if lr_shedule:
                    lr_sheduler.step()

                all_results[trial_name]['training_loss'].append(loss.to('cpu').detach().float())

                model.eval()
                test_loss = 0
                avg_gap = 0
                avg_error = 0
                avg_recall = 0
                for data in test_data:
                    if edge_attributes:
                        out = model(
                            x = data.x.to(device), 
                            edge_index=data.edges.to(device), 
                            edge_attr=data.edge_features.to(device)
                        )
                    else:
                        out = model(
                            x = data.x.to(device), 
                            edge_index=data.edges.to(device), 
                        )  
                    pred_k = F.softmax(pred, dim=1).argmax(axis=1)
                    test_loss += F.cross_entropy(pred, data.k_labels.to(device))
                    gap, error = calculate_gap(data.P, data.R, data.V, pred_k, K, device='cpu')
                    avg_gap += gap
                    avg_error += error
                    avg_recall += recall_score(data.k_labels, pred_k.to('cpu'), average="macro")


                test_loss /= len(test_data)
                avg_gap /= len(test_data)
                avg_error /= len(test_data)
                avg_recall /= len(test_data)

                all_results[trial_name]['test_loss'].append(test_loss.to('cpu').detach().float())
                all_results[trial_name]['test_gap'].append(avg_gap)
                all_results[trial_name]['test_error'].append(avg_error)
                all_results[trial_name]['test_recall'].append(avg_recall)

                epochs.set_description(f"Trial {trial_num}, Epoch {epoch+1}/{N_epochs}, Train Loss {loss:.4f} Test Loss {test_loss:.4f}, Gap {avg_gap:.4f}, Recall {avg_recall:.4f}")

                if early_stopping:
                    if test_loss < best_loss:
                        early_stopping_counter = 0
                        best_loss = test_loss
                    else:
                        early_stopping_counter += 1
                    
                    if early_stopping_counter > early_stopping_patience:
                        min_epochs = epoch
                        break

                    if epoch >= min_epochs:
                        break


        if early_stopping:
            for trial_name in all_results.keys():
                for key in all_results[trial_name].keys():
                    all_results[trial_name][key] = all_results[trial_name][key][:min_epochs]

        print("Saving Data to files")
        processed = {}
        for key in all_results["trial_0"].keys():
            df = pd.DataFrame({trial_id:all_results[trial_id][key] for trial_id in all_results.keys()}).astype(float)
            df.to_csv(f"Results/{experiment_name}/{filename}/{key}.csv")
            processed[key] = df
        print("Generating plots")
        n_plots = len(all_results["trial_0"].keys())
        n_cols = 2
        n_rows = int((n_plots + n_plots%2)/2)

        fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, 6*n_rows))

        count = 0
        for key in all_results["trial_0"].keys():
            col = count %2
            row = count //2 
            count += 1

            df_long = processed[key].stack()
            df_long.index = df_long.index.to_flat_index().map(lambda x: x[0])
            sns.lineplot(df_long, errorbar='ci', ax=ax[row, col])
            ax[row, col].set_ylabel(key)
            ax[row, col].set_xlabel("Epoch")

        plt.savefig(f"Results/{experiment_name}/{filename}/plots.png")

print("Done!")



# %%
