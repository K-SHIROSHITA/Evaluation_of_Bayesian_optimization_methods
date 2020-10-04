import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import zscore

from botorch.models.pairwise_gp import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import qNoisyExpectedImprovement, ExpectedImprovement, UpperConfidenceBound
from botorch.optim import optimize_acqf
import torch
import gpytorch

from accuracy.create_function import generate_data, generate_comparisons
from accuracy.self_acquisition import MaxVariance

def main(n, m, dim, noise, bounds):
    train_X_random, train_X_mesh, train_y_random, train_y_mesh = generate_data(n, dim=dim)

    train_comp_random = generate_comparisons(train_y_random, m, noise=noise)

    covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=dim))
    model = PairwiseGP(train_X_random, train_comp_random, covar_module=covar_module)
    mll = PairwiseLaplaceMarginalLogLikelihood(model)
    mll = fit_gpytorch_model(mll)

    qNEI = qNoisyExpectedImprovement(
        model=model,
        X_baseline=train_X_random.float()
    )

    EI = ExpectedImprovement(
        model=model,
        best_f=0.9
    )

    UCB = UpperConfidenceBound(
        model=model,
        beta=1
    )
    MV = MaxVariance(
        model=model,
        beta=1
    )
    
    next_X, acq_val = optimize_acqf(
        acq_function=qNEI,
        bounds=bounds,
        q=2,
        num_restarts=5,
        raw_samples=256
    )

    points = torch.linspace(-1, 1, 101)
    pred_y = zscore(model.posterior(points).mean.squeeze().detach().numpy())
    train_y_mesh = zscore(train_y_mesh)

    mse = np.square(train_y_mesh - pred_y).mean()

    plt.clf()
    plt.plot(train_X_mesh, train_y_mesh)
    plt.plot(points, pred_y)

    plt.show()
    print("MSE", mse)
    print(next_X)
    print(acq_val)


if __name__ == '__main__':
    N = 100
    M = 50
    DIM = 1
    NOISE = 0.1
    BOUNDS = torch.stack([torch.zeros(DIM)-1,
                          torch.ones(DIM)])

    main(n=N, m=M, dim=DIM, noise=NOISE, bounds=BOUNDS)
