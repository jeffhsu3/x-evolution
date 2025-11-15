from __future__ import annotations
from typing import Callable

import torch
from torch import tensor
from torch.nn import Module
import torch.nn.functional as F
from torch.func import functional_call, vmap

from beartype import beartype
from beartype.door import is_bearable

from accelerate import Accelerator

from x_mlps_pytorch.noisable import (
    Noisable,
    with_seed
)

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def normalize(t, eps = 1e-6):
    return F.layer_norm(t, t.shape[-1:], eps = eps)

# class

class EvoStrategy(Module):

    @beartype
    def __init__(
        self,
        model: Module,
        *,
        environment: Callable[[Module], float],  # the environment is simply a function that takes in the model and returns a fitness score
        num_generations,
        population_size = 30,
        param_names_to_optimize: list[str] | None = None,
        fitness_to_weighted_factor: Callable[[Tensor], Tensor] = normalize,
        cpu = False,
        accelerate_kwargs: dict = dict(),
    ):
        super().__init__()

        self.accelerate = Accelerator(cpu = cpu, **accelerate_kwargs)

        self.model = model
        self.noisable_model = Noisable(model)

        self.environment = environment

        param_names = set(dict(model.named_parameters()).keys())

        # default to all parameters to optimize with evo strategy

        param_names_to_optimize = default(param_names_to_optimize, param_names)

        # validate

        assert all([name in param_names for name in param_names_to_optimize])
        assert len(param_names_to_optimize) > 0, 'nothing to optimize'

        # sort param names and store

        param_names_list = list(param_names_to_optimize)
        param_names_list.sort()

        self.param_names_to_optimize = param_names_list

        # hyperparameters

        self.population_size = population_size
        self.num_params = len(param_names_list) # just convenience for generating all the seeds for all the randn for the proposed memory efficient way

        self.num_generations = num_generations

        # the function that transforms a tensor of fitness floats to the weight for the weighted average of the noise for rolling out 1x1 ES

        self.fitness_to_weighted_factor = fitness_to_weighted_factor

        self.register_buffer('_dummy', tensor(0), persistent = False)

    @property
    def device(self):
        return self._dummy.device

    def evolve_(
        self,
        fitnesses: list[float] | Tensor
    ):
        if isinstance(fitnesses, list):
            fitnesses = tensor(fitnesses)

        fitnesses = fitnesses.to(self.device)

        # they use a simple z-score for the fitnesses, need to figure out the natural ES connection

        noise_weights = self.fitness_to_weighted_factor(fitnesses)

    def forward(
        self
    ):

        fitnesses = []

        for _ in range(self.num_generations):

            with self.noisable_model.temp_add_noise_(dict()):
                fitness = self.environment(noisable_model)

            fitnesses.append(fitness)

        self.evolve_(fitnesses)
