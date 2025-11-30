import pytest
param = pytest.mark.parametrize

import torch
from x_mlps_pytorch import MLP

model = MLP(8, 16, 4)

@param('params_to_optimize', (None, ['layers.1.weight'], [model.layers[1].weight]))
@param('use_optimizer', (False, True))
@param('noise_low_rank', (None, 1))
@param('mirror_sampling', (False, True))
def test_evo_strat(
    params_to_optimize,
    use_optimizer,
    noise_low_rank,
    mirror_sampling
):
    from random import randrange

    from x_evolution.x_evolution import EvoStrategy

    evo_strat = EvoStrategy(
        model,
        environment = lambda model: float(randrange(100)),
        num_generations = 1,
        params_to_optimize = params_to_optimize,
        use_optimizer = use_optimizer,
        noise_low_rank = noise_low_rank,
        mirror_sampling = mirror_sampling
    )

    evo_strat('evolve')
    evo_strat('more.evolve')
