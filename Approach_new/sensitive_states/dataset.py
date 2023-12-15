from typing import List, Tuple, Union
import torch
from torch_geometric.data import Dataset, InMemoryDataset
import os
import json
from torch_geometric.data import Data
import numpy as np

class MDPDataset(Dataset):
    def __init__(self, root, pre_transform, raw_dir = "raw", processed_dir="processed", transform=None):
        if not os.path.isdir(os.path.join(root, "processed")):
            os.mkdir(os.path.join(root, "processed"))

        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)
    
    @property
    def processed_file_names(self):
        return os.listdir(self.processed_dir)

    def len(self):
        files = os.listdir(self.raw_dir)
        return len(files)
    
    def process(self):
        for i in range(self.len()):
            filename = os.path.join(self.raw_dir, f"mdp_{i}.json")
            raw_data = json.load(open(filename, 'r'))

            P = torch.tensor(raw_data['transitions'], dtype=torch.float32)
            R = torch.tensor(raw_data['rewards'], dtype=torch.float32)
            V = torch.tensor(raw_data['optimal_values'], dtype=torch.float32)
            k_states = torch.tensor(raw_data['k_states'], dtype=torch.int64)

            data = self.pre_transform(P, R, V, k_states)
            torch.save(data, os.path.join(self.processed_dir, f"mdp_{i}.pt"))
    
    def get(self, idx):
        filename = os.path.join(self.processed_dir, f"mdp_{idx}.pt")
        data = torch.load(filename)
        return data
    
    # def __get__(self, idx):
    #     return self.get(idx)
    
class InMemoryMDPDataset:
    def __init__(self, root, pre_transform, transform=None, pre_filter=None):
        self.raw_dir=os.path.join(root, "raw")
        self.processed_dir = os.path.join(root, "processed")
        self.pre_transform = pre_transform
        
        self.dataset = []

        self.process()

    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        return os.listdir(self.processed_dir)
    
    def __len__(self):
        files = os.listdir(self.raw_dir)
        return len(files)
    
    def len(self):
        return len(self)

    def process(self):
        for i in range(self.len()):
            filename = os.path.join(self.raw_dir, f"mdp_{i}.json")
            raw_data = json.load(open(filename, 'r'))

            P = torch.tensor(raw_data['transitions'], dtype=torch.float32)
            R = torch.tensor(raw_data['rewards'], dtype=torch.float32)
            V = torch.tensor(raw_data['optimal_values'], dtype=torch.float32)
            k_states = torch.tensor(raw_data['k_states'], dtype=torch.int64)

            data = self.pre_transform(P, R, V, k_states)
            self.dataset.append(data)

    def __getitem__(self, idx):
        return self.dataset[idx]

class AllNodeFeatures:
    def __init__(self):
        pass

    def __call__(self, P, R, V, k_states):
        N_states, N_actions = R.shape
    
        T = torch.empty((N_states, N_states*N_actions))
        for i in range(N_states):
            T[i, :] = P[:, i, :].reshape(1, -1)

        x = torch.cat([T, R], dim=1)

        # Cound whether transition are non-zero for any action
        p_sum = torch.sum((P> 0), axis=0)
        edges = torch.tensor([[i, j] for i, j in zip(*torch.where(p_sum > 0))])

        data = Data(
            x=x,
            edges=edges.T,
            edge_features=None,
            k_labels=k_states,
            P = P,
            R = R,
            V = V
        )
        return data

class TransitionsOnEdge:
    def __init__(self):
        pass

    def __call__(self, P, R, V, k_states):
        p_sum = torch.sum((P> 0), axis=0)
        edges = torch.tensor([[i, j] for i, j in zip(*torch.where(p_sum > 0))])
        edge_features = torch.Tensor([P[:, i[0], i[1]].numpy() for i in edges])

        data = Data(
                    x=R,
                    edges=edges.T,
                    edge_features = edge_features,
                    k_labels=k_states,
                    P = P,
                    R = R,
                    V = V
                )
        return data