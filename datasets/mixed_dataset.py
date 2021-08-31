"""
This file contains the definition of different heterogeneous datasets used for training
"""
import torch
import numpy as np

from .base_dataset import BaseDataset

class MixedDataset(torch.utils.data.Dataset):

    def __init__(self, options, **kwargs):
        self.dataset_list = ['lsp-orig', 'mpii', 'lspet', 'coco', 'mpi-inf-3dhp']
        self.dataset_dict = {'lsp-orig': 0, 'mpii': 1, 'lspet': 2, 'coco': 3, 'mpi-inf-3dhp': 4}
        self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
        total_length = sum([len(ds) for ds in self.datasets])
        length_itw = sum([len(ds) for ds in self.datasets[0:-1]])
        self.length = max([len(ds) for ds in self.datasets])
        
        self.partition = [.6*len(self.datasets[0])/length_itw, 
                        .6*len(self.datasets[1])/length_itw, 
                        .6*len(self.datasets[2])/length_itw, 
                        .6*len(self.datasets[3])/length_itw, 
                        0.4 ]
        self.partition = np.array(self.partition).cumsum()
        # print(self.partition)

    def __getitem__(self, index):
        p = np.random.rand()
        for i in range(len(self.dataset_list)):    
            if p <= self.partition[i]:
                # print(i)
                return self.datasets[i][index % len(self.datasets[i])]

    def __len__(self):
        return self.length
