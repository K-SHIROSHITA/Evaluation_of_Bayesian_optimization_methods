import os
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore

from botorch.models.pairwise_gp import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import qNoisyExpectedImprovement
from botorch.optim import optimize_acqf
import torch

from create_function import generate_data, generate_comparisons, utility

IMAGE_DIR = "./qNEI1_qNEI2/image"
if not os.path.exists(IMAGE_DIR):
    # ディレクトリが存在しない場合、ディレクトリを作成する
    os.makedirs(IMAGE_DIR)


def observation_max_points(model, results):
    observation_point = model.posterior(results).mean.tolist()
    next_x_index = observation_point.index(max(observation_point))
    next_x = results[next_x_index]

    return next_x


def plot_6d(x, y, image_name):
    plt.clf()
    y_2D_12 = y.mean(axis=5).mean(axis=4).mean(axis=3).mean(axis=2)
    plt.contourf(x, x, y_2D_12)
    plt.savefig(image_name + "_12.png")
    plt.clf()
    y_2D_34 = y.mean(axis=5).mean(axis=4).mean(axis=1).mean(axis=0)
    plt.contourf(x, x, y_2D_34)
    plt.savefig(image_name + "_34.png")
    plt.clf()
    y_2D_56 = y.mean(axis=3).mean(axis=2).mean(axis=1).mean(axis=0)
    plt.contourf(x, x, y_2D_56)
    plt.savefig(image_name + "_56.png")


def main(n, m, dim, noise, bounds):
    train_X, train_y = generate_data(n, dim=dim)
    train_comp = generate_comparisons(train_y, m, noise=noise)

    model = PairwiseGP(train_X, train_comp)
    mll = PairwiseLaplaceMarginalLogLikelihood(model)
    mll = fit_gpytorch_model(mll)

    points = torch.linspace(0, 1, 11)
    points_0, points_1, points_2, points_3, points_4, points_5 = torch.meshgrid(points, points, points, points, points,
                                                                                points)
    points_mesh = torch.stack((points_0, points_1, points_2, points_3, points_4, points_5), dim=-1)
    pred_y_std = zscore(model.posterior(points_mesh).mean.squeeze().detach().numpy())
    pred_y = model.posterior(points_mesh).mean.squeeze().detach().numpy()
    val_y = zscore(utility(points_mesh)).reshape(11, 11, 11, 11, 11, 11)

    mse_list = []
    mse = np.square(val_y - pred_y_std).mean()
    mse_list.append(mse)
    print("MSE", mse)

    # 検証用の関数(2次元だけ取り出してあとは平均値)
    image_name = "./qNEI1_qNEI2/image/val_2D"
    plot_6d(points, val_y, image_name=image_name)

    # 予測平均をplot(2次元だけ取り出してあとは平均値)
    image_name = "./qNEI1_qNEI2/image/pred_y_mean_2D_0"
    plot_6d(points, pred_y, image_name=image_name)

    for i in range(10):
        qNEI = qNoisyExpectedImprovement(
            model=model,
            X_baseline=train_X.float()
        )

        next_X, acq_val = optimize_acqf(
            acq_function=qNEI,
            bounds=bounds,
            q=2,
            num_restarts=5,
            raw_samples=256
        )

        next_X = torch.cat((next_X[0].reshape(-1, 6), next_X[1].reshape(-1, 6)), dim=0).type(torch.float64)
        train_X = torch.cat((train_X, next_X), dim=0)

        if utility(next_X[0]) > utility(next_X[1]):
            preference = torch.Tensor([len(train_X) - 2, len(train_X) - 1]).long().reshape(-1, 2)
            train_comp = torch.cat((train_comp, preference), dim=0)
        else:
            preference = torch.Tensor([len(train_X) - 1, len(train_X) - 2]).long().reshape(-1, 2)
            train_comp = torch.cat((train_comp, preference), dim=0)

        model = PairwiseGP(train_X, train_comp)
        mll = PairwiseLaplaceMarginalLogLikelihood(model)
        mll = fit_gpytorch_model(mll)
        pred_y_std = zscore(model.posterior(points_mesh).mean.squeeze().detach().numpy())
        pred_y = model.posterior(points_mesh).mean.squeeze().detach().numpy()

        mse = np.square(val_y - pred_y_std).mean()
        mse_list.append(mse)
        print("MSE", mse)

        # 予測平均をplot(2次元だけ取り出してあとは平均値)
        image_name = "./qNEI1_qNEI2/image/pred_y_mean_2D_" + str(i + 1)
        plot_6d(points, pred_y, image_name=image_name)

    number_of_comparisons = [m]
    for i in range(len(mse_list) - 1):
        number_of_comparisons.append(m + i)

    plt.clf()
    plt.plot(number_of_comparisons, mse_list)
    plt.savefig("./qNEI1_qNEI2/image/mse.png")
