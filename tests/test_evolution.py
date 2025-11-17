import pytest
param = pytest.mark.parametrize

import torch

@param('param_names_to_optimize', (None, ['layers.1.weight']))
def test_evo_strat(
    param_names_to_optimize
):
    from random import randrange

    from x_evolution.x_evolution import EvoStrategy

    from x_mlps_pytorch import MLP
    model = MLP(8, 16, 4)

    evo_strat = EvoStrategy(
        model,
        environment = lambda model: float(randrange(100)),
        num_generations = 10,
        param_names_to_optimize = param_names_to_optimize
    )

    evo_strat()
