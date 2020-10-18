import warnings

warnings.filterwarnings("ignore")

from itertools import combinations
import numpy as np
import torch

from botorch.test_functions import Hartmann




def utility(X):
    """ Given X, output corresponding utility (i.e., the latent function)
    """
    f_test = Hartmann(dim=6)
    X = X.reshape(-1, 6)
    y = f_test.evaluate_true(X)
    return y


def generate_data(n, dim=6):
    """ Generate data X and y """
    # X is randomly sampled from dim-dimentional unit cube
    # we recommend using double as opposed to float tensor here for
    # better numerical stability
    X = torch.rand(n, dim, dtype=torch.float64)  # 書き換えてる

    y = utility(X)

    return X, y


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
