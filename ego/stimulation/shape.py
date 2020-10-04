from os import path, remove
from math import sin, cos

import numpy as np
import matplotlib.pyplot as plt
import torch
import gpytorch

from botorch.models import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound
from botorch.fit import fit_gpytorch_model
from botorch.optim import optimize_acqf


# from ego.acquisition import MaxVariance


def random_gallery(bounds):
    tmp_gallery = []
    for i in range(len(bounds[0])):
        tmp_gallery.append(np.random.uniform(bounds[0][i], bounds[1][i]))
        tmp_gallery.append(np.random.uniform(bounds[0][i], bounds[1][i]))
    tmp_gallery = np.array(tmp_gallery)
    tmp_gallery = tmp_gallery.reshape(2, len(bounds[0]))
    tmp_gallery = tmp_gallery.tolist()
    return tmp_gallery


def observation_max_points(results, responses, bounds):
    results = torch.Tensor(results).reshape(-1, len(bounds[0]))
    responses = torch.LongTensor(responses).reshape(-1, 2)
    covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=len(bounds[0])))
    model = PairwiseGP(results, responses, covar_module=covar_module, std_noise=0.1)
    mll = PairwiseLaplaceMarginalLogLikelihood(model)
    mll = fit_gpytorch_model(mll)

    observation_point = model.posterior(results, observation_noise=True).mean.tolist()
    next_x_index = observation_point.index(max(observation_point))
    next_x = results[next_x_index]

    return next_x


def acquisition_gallery(results, responses, bounds, q=1, num_restarts=5, num_raw_samples=20):
    results = torch.Tensor(results).reshape(-1, len(bounds[0]))
    responses = torch.LongTensor(responses).reshape(-1, 2)
    covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=len(bounds[0])))
    model = PairwiseGP(results, responses, covar_module=covar_module, std_noise=0.1)
    mll = PairwiseLaplaceMarginalLogLikelihood(model)
    mll = fit_gpytorch_model(mll)

    # acq_func = MaxVariance(model, beta=1)

    acq_func = ExpectedImprovement(model, best_f=1.0)

    # beta = np.sqrt(np.log(len(results) / 2) / (len(results) / 2))
    # acq_func = UpperConfidenceBound(model, beta=1.0)

    next_x, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=q,
        num_restarts=num_restarts,
        raw_samples=num_raw_samples
    )

    prev_result_max = observation_max_points(results, responses, bounds)
    acq_galleries = torch.cat((next_x.reshape(-1, len(bounds[0])), prev_result_max.reshape(-1, len(bounds[0]))))
    acq_galleries = acq_galleries.reshape(-1, len(bounds[0]))
    acq_galleries = acq_galleries.tolist()
    return acq_galleries


def image_create_ellipse_fourier(i_path, r, results):
    if path.isfile(i_path) is True:
        remove(i_path)

    results.append(r)
    x = []
    y = []
    s = 0.01
    while s <= 6.29:
        x.append(sin(s) * r[0] + sin(2 * s) * r[1] + sin(3 * s) * r[2])
        y.append(cos(s) * r[3] + cos(2 * s) * r[4] + cos(3 * s) * r[5])

        s += 0.01

    plt.clf()
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)
    plt.gca().spines["bottom"].set_visible(False)
    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    plt.axis("off")
    # -----plot setting end-----

    # plot image
    plt.plot(x, y)
    # save image
    plt.savefig(i_path)


def image_create_ellipse_fourier_debug(i_path, r):
    if path.isfile(i_path) is True:
        remove(i_path)

    x = []
    y = []
    s = 0.01
    while s <= 6.29:
        x.append(sin(s) * r[0] + sin(2 * s) * r[1] + sin(3 * s) * r[2])
        y.append(cos(s) * r[3] + cos(2 * s) * r[4] + cos(3 * s) * r[5])
        s += 0.01
    plt.clf()
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)
    plt.gca().spines["bottom"].set_visible(False)
    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    plt.axis("off")
    # -----plot setting end-----

    # plot image
    plt.plot(x, y)
    # save image
    plt.savefig(i_path)
