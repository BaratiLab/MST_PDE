# -*- coding: utf-8 -*-
"""
@author: AmirPouya Hemmasian (ahemmasi@andrew.cmu.edu)
"""
import torch
from torch.utils.data import Dataset


class dataset2d(Dataset):
    def __init__(self, data_array, idxs):
        """
        The shape of data_array can be (T, H, W) or (N, T, H, W)
        """
        super().__init__()
        self.data = torch.as_tensor(data_array, dtype=torch.float)
        self.idxs = idxs
        if len(self.data.shape) == 3:
            # if shape is (T, H, W), make it (N, T, H, W) where N=1
            self.data = self.data.unsqueeze(0)
        # add channel dimension (N, T, C, H, W)
        self.data = self.data.unsqueeze(2)
        self.dt = 1
        self.rollout = 0
        self.N = len(idxs)
        self.T = self.data.shape[1]
        self.hashmap = lambda i: (i//self.T, i%self.T)

    def __len__(self):
        return self.N*(self.T - self.rollout*self.dt)

    def __getitem__(self, idx):
        n, t = self.hashmap(idx)
        out = [self.data[self.idxs[n], t+i*self.dt] for i in range(self.rollout+1)]
        return out

    def get_item(self, n, t):
        return self.data[self.idxs[n], t]

    def config(self, dt=1, rollout=0):
        self.dt = min(dt, self.T-1)
        self.rollout = min(rollout, (self.T-1)//self.dt)
        self.hashmap = lambda i: (i//(self.T-self.rollout*self.dt),
                                  i%(self.T-self.rollout*self.dt))
        return self.rollout


class dataset1d(Dataset):
    def __init__(self, data_array, idxs):
        """
        The shape of data_array can be (T, H) or (N, T, H)
        """
        super().__init__()
        self.data = torch.as_tensor(data_array, dtype=torch.float)
        self.idxs = idxs
        if len(self.data.shape) == 2:
            # if shape is (T, H), make it (N, T, H) where N=1
            self.data = self.data.unsqueeze(0)
        # add channel dimension (N, T, C, H)
        self.data = self.data.unsqueeze(2)
        self.dt = 1
        self.rollout = 0
        self.N = len(idxs)
        self.T = self.data.shape[1]
        self.hashmap = lambda i: (i//self.T, i%self.T)

    def __len__(self):
        return self.N*(self.T - self.rollout*self.dt)

    def __getitem__(self, idx):
        n, t = self.hashmap(idx)
        out = [self.data[self.idxs[n], t+i*self.dt] for i in range(self.rollout+1)]
        return out

    def get_item(self, n, t):
        return self.data[self.idxs[n], t]

    def config(self, dt=1, rollout=0):
        self.dt = min(dt, self.T-1)
        self.rollout = min(rollout, (self.T-1)//self.dt)
        self.hashmap = lambda i: (i//(self.T-self.rollout*self.dt),
                                  i%(self.T-self.rollout*self.dt))
        return self.rollout

