import warnings

warnings.filterwarnings("ignore")
from itertools import combinations
import numpy as np
import torch
import gpytorch


def utility(X, covar_module):
    """ Given X, output corresponding utility (i.e., the latent function)
    """
    y = X[0]

    return y


def generate_data(n, dim=1):
    # ガウス過程に従う関数を作成する
    covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=1))
    """ Generate data X and y """
    # X is randomly sampled from dim-dimentional unit cube
    # we recommend using double as opposed to float tensor here for
    # better numerical stability
    X_random = torch.rand(n, dim, dtype=torch.float64) * 2 - 1  # 書き換えてる
    X_mesh = torch.linspace(-1, 1, 101).reshape(-1, 1)
    X_all = torch.Tensor(np.vstack((X_random.numpy(), X_mesh.numpy())))
    y_all = utility(X_all, covar_module)
    y_random = y_all[:n]
    y_mesh = y_all[n:]

    return X_random, X_mesh, y_random, y_mesh


def generate_comparisons(y, n_comp, replace=False, noise=None):
    """  Create pairwise comparisons with noise """
    # generate all possible pairs of elements in y
    all_pairs = np.array(list(combinations(range(y.shape[0]), 2)))
    # randomly select n_comp pairs from all_pairs
    comp_pairs = all_pairs[np.random.choice(range(len(all_pairs)), n_comp, replace=replace)]
    # add gaussian noise to the latent y values
    c0 = y[comp_pairs[:, 0]] + np.random.standard_normal(len(comp_pairs)) * noise
    c1 = y[comp_pairs[:, 1]] + np.random.standard_normal(len(comp_pairs)) * noise
    reverse_comp = (c0 < c1)
    comp_pairs[reverse_comp, :] = np.flip(comp_pairs[reverse_comp, :], 1)
    comp_pairs = torch.tensor(comp_pairs).long()

    return comp_pairs
