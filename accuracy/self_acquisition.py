from __future__ import annotations

from abc import ABC
from typing import Optional, Union

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.objective import ScalarizedObjective
from botorch.exceptions import UnsupportedError

from botorch.models.model import Model
from botorch.posteriors.posterior import Posterior

from botorch.utils.transforms import convert_to_target_pre_hook, t_batch_mode_transform
from torch import Tensor


class AnalyticAcquisitionFunction(AcquisitionFunction, ABC):
    r"""Base class for analytic acquisition functions."""

    def __init__(
            self, model: Model, objective: Optional[ScalarizedObjective] = None
    ) -> None:
        r"""Base constructor for analytic acquisition functions.

        Args:
            model: A fitted single-outcome model.
            objective: A ScalarizedObjective (optional).
        """
        super().__init__(model=model)
        if objective is None:
            if model.num_outputs != 1:
                raise UnsupportedError(
                    "Must specify an objective when using a multi-output model."
                )
        elif not isinstance(objective, ScalarizedObjective):
            raise UnsupportedError(
                "Only objectives of type ScalarizedObjective are supported for "
                "analytic acquisition functions."
            )
        self.objective = objective

    def _get_posterior(self, X: Tensor) -> Posterior:
        r"""Compute the posterior at the input candidate set X.

        Applies the objective if provided.

        Args:
            X: The input candidate set.

        Returns:
            The posterior at X. If a ScalarizedObjective is defined, this
            posterior can be single-output even if the underlying model is a
            multi-output model.
        """
        posterior = self.model.posterior(X)
        if self.objective is not None:
            # Unlike MCAcquisitionObjective (which transform samples), this
            # transforms the posterior
            posterior = self.objective(posterior)
        return posterior

    def set_X_pending(self, X_pending: Optional[Tensor] = None) -> None:

        raise UnsupportedError(
            "Analytic acquisition functions do not account for X_pending yet."
        )


class MaxVariance(AnalyticAcquisitionFunction):

    def __init__(
            self,
            model: Model,
            beta: Union[float, Tensor],
            objective: Optional[ScalarizedObjective] = None,
            maximize: bool = True,
    ) -> None:

        super().__init__(model=model, objective=objective)
        self.maximize = maximize
        if not torch.is_tensor(beta):
            beta = torch.tensor(beta)
        self.register_buffer("beta", beta)

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        self.beta = self.beta.to(X)
        posterior = self._get_posterior(X=X)
        batch_shape = X.shape[:-2]
        variance = posterior.variance.view(batch_shape)
        delta = variance
        if self.maximize:
            return delta
        else:
            return delta
