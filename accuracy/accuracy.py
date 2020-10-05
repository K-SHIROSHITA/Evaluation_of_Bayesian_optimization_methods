import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore

from botorch.models.pairwise_gp import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import qNoisyExpectedImprovement, ExpectedImprovement, UpperConfidenceBound
from botorch.optim import optimize_acqf
import torch
import gpytorch

from create_function import generate_data, generate_comparisons
from self_acquisition import MaxVariance

IMAGE_DIR = "image"
if not os.path.exists(IMAGE_DIR):
    # ディレクトリが存在しない場合、ディレクトリを作成する
    os.makedirs(IMAGE_DIR)


def observation_max_points(results, responses, bounds):
    results = torch.Tensor(results).reshape(-1, len(bounds[0]))
    responses = torch.LongTensor(responses).reshape(-1, 2)
    covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=len(bounds[0])))
    model = PairwiseGP(results, responses, covar_module=covar_module, std_noise=0.1)
    mll = PairwiseLaplaceMarginalLogLikelihood(model)
    mll = fit_gpytorch_model(mll)

    observation_point = model.posterior(results).mean.tolist()
    next_x_index = observation_point.index(max(observation_point))
    next_x = results[next_x_index]

    return next_x


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

    next_X0, acq_val = optimize_acqf(
        acq_function=EI,
        bounds=bounds,
        q=1,
        num_restarts=5,
        raw_samples=256
    )

    next_X1 = observation_max_points(results=train_X_random.float(), responses=train_y_random, bounds=bounds)

    points = torch.linspace(-1, 1, 101)
    pred_y = zscore(model.posterior(points).mean.squeeze().detach().numpy())
    train_y_mesh = zscore(train_y_mesh)

    mse = np.square(train_y_mesh - pred_y).mean()

    plt.clf()
    plt.plot(train_X_mesh, train_y_mesh)
    plt.plot(points, pred_y)
    plt.savefig("./image/function_plot.png")
    
    print("MSE", mse)
    print("次の探索候補の座標", next_X0)
    print("探索済の最大予測平均の座標", next_X1)


if __name__ == '__main__':
    N = 100
    M = 50
    DIM = 1
    NOISE = 0.1
    BOUNDS = torch.stack([torch.zeros(DIM) - 1,
                          torch.ones(DIM)])

    main(n=N, m=M, dim=DIM, noise=NOISE, bounds=BOUNDS)
