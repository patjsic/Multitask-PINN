import os
import numpy as np
import scipy
import torch
from torch.utils.data import Dataset

"""
Data code mostly taken and adapted from: https://github.com/chen-yingfa/pinn-torch/blob/master/data.py.
"""

# class MTLDataset(Dataset):
#     """
#     Dataset that only returns the data of a specifc task (i.e., only data or collocation)
#     """
#     def __init__(self, data, collocation=False):
#         self.data = data
#         self.collocation = collocation
#         self.examples = torch.tensor(data, dtype=torch.float32, requires_grad=True)
    
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         if self.collocation:
#             headers = ["t", "x", "y",]


class PINNDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.examples = torch.tensor(data, dtype=torch.float32, requires_grad=True)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        headers = ["t", "x", "y", "p", "u", "v"]
        return {key: self.examples[idx, i] for i, key in enumerate(headers)}

def get_navier_data(data_dir, noise=0.1):
    """
    Load and preprocesses 2D cylindrical Navier Stokes data. 
    Returns train dataset, test dataset, and bounds for x.
    """
    np.random.seed(42)
    data_path = os.path.join(data_dir, "cylinder_wake.mat")
    data = scipy.io.loadmat(data_path)

    #Separate data into arrays
    X_star = data["X_star"] # N X 2
    x = X_star[:,0:1]
    y = X_star[:,0:1]

    U_star = data["U_star"] # N X 2 X T
    P_star = data["p_star"] # N X T
    t_star = data["t"] # T X 1
    X_star = data["X_star"] # N X 2

    N = X_star.shape[0]
    T = t_star.shape[0]

    #Reshape Data
    XX = np.tile(X_star[:,0:1], (1, T)) # N X T
    YY = np.tile(X_star[:,1:2], (1, T)) # N X T
    TT = np.tile(t_star, (1,N)).T # N X T
    
    UU = U_star[:, 0, :] # N X T
    VV = U_star[:, 1, :] # N X T
    PP = P_star # N X T

    #Flatten to make data all same shape NT X 1
    x = XX.flatten()[:, None]
    y = YY.flatten()[:, None]
    t = TT.flatten()[:, None]

    u = UU.flatten()[:, None]
    v = VV.flatten()[:, None]
    p = PP.flatten()[:, None]

    min_x = np.min(x)
    max_x = np.max(x)

    #Add Gaussian noise to data
    u += noise * np.std(u) * np.random.randn(*u.shape)
    v += noise * np.std(v) * np.random.randn(*v.shape)

    train_data = np.hstack((t, x, y, p, u, v))

    #Sample 1000 points as test data
    idx = np.random.choice(train_data.shape[0], train_data.shape[0] - 1000, replace=False)
    test_data = np.delete(train_data, idx, axis=0) #Remove points from train data that are selected for test data
    train_idx = np.random.choice(train_data.shape[0], 10000, replace=False) #For debugging subsample training data
    train_data = train_data[train_idx, :] #For debugging subselect training data

    print(train_data.shape)
    print(test_data.shape)

    train_data = PINNDataset(train_data)
    test_data = PINNDataset(test_data)

    return train_data, test_data, min_x, max_x

if __name__ == "__main__":
    train_dataset, test_dataset, min_val, max_val = get_navier_data("data")