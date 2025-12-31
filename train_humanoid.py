# /// script
# dependencies = [
#     "gymnasium[mujoco]>=1.0.0",
#     "gymnasium[other]",
#     "x-evolution>=0.0.20",
#     "x-mlps-pytorch"
# ]
# ///

# import os
# os.environ["NCCL_P2P_DISABLE"] = "1"
# os.environ["NCCL_IB_DISABLE"] = "1"
# os.environ["MUJOCO_GL"] = "osmesa"

from shutil import rmtree
import gymnasium as gym
import numpy as np

import torch
from torch.nn import Module, GRU, Linear
import torch.nn.functional as F

# functions

def exists(v):
    return v is not None

def softclamp(t, value):
    return (t / value).tanh() * value

class HumanoidEnvironment(Module):
    def __init__(
        self,
        video_folder = './recordings_humanoid',
        render_every_eps = 100,
        max_steps = 1000,
        repeats = 1
    ):
        super().__init__()

        # Humanoid-v5
        env = gym.make('Humanoid-v5', render_mode = 'rgb_array')

        self.env = env
        self.max_steps = max_steps
        self.repeats = repeats
        self.video_folder = video_folder
        self.render_every_eps = render_every_eps

    def pre_main_callback(self):
        # the `pre_main_callback` on the environment passed in is called before the start of the evolutionary strategies loop

        rmtree(self.video_folder, ignore_errors = True)

        self.env = gym.wrappers.RecordVideo(
            env = self.env,
            video_folder = self.video_folder,
            name_prefix = 'recording',
            episode_trigger = lambda eps_num: (eps_num % self.render_every_eps) == 0,
            disable_logger = True
        )

    def forward(self, model):

        device = next(model.parameters()).device

        seed = torch.randint(0, int(1e6), ())

        cum_reward = 0.

        for _ in range(self.repeats):
            state, _ = self.env.reset(seed = seed.item())

            step = 0
            hiddens = None
            last_action = None
            
            while step < self.max_steps:

                state = torch.from_numpy(state).float().to(device)

                action_logits, hiddens = model(state, hiddens)

                mean, log_var = action_logits.chunk(2, dim = -1)

                # sample and then bound and scale to -0.4 to 0.4

                std = (0.5 * softclamp(log_var, 5.)).exp()
                sampled = mean + torch.randn_like(mean) * std
                action = sampled.tanh() * 0.4

                next_state, reward, truncated, terminated, info = self.env.step(action.detach().cpu().numpy())

                # reward functions

                # encouraged to move forward (1.0) and stay upright (> 1.2 meters)

                z_pos = next_state[0]
                x_vel = next_state[5]

                reward_forward = x_vel
                reward_upright = float(z_pos > 1.2)

                exploration_bonus = std.mean() * 0.05
                penalize_extreme_actions = (mean.abs() > 1.).float().mean() * 0.05

                penalize_action_change = 0.
                if exists(last_action):
                    penalize_action_change = (last_action - action).abs().mean() * 0.1

                cum_reward += float(reward) + reward_forward + reward_upright + exploration_bonus - penalize_extreme_actions - penalize_action_change

                step += 1

                state = next_state
                last_action = action

                if truncated or terminated:
                    break

        return cum_reward / self.repeats

# evo strategy

from x_evolution import EvoStrategy

from x_mlps_pytorch.residual_normed_mlp import ResidualNormedMLP

class Model(Module):

    def __init__(self):
        super().__init__()

        self.deep_mlp = ResidualNormedMLP(
            dim_in = 348,
            dim = 256,
            depth = 8,
            residual_every = 2
        )

        self.gru = GRU(256, 256, batch_first = True)

        self.to_pred = Linear(256, 17 * 2, bias = False)

    def forward(self, state, hiddens = None):

        x = self.deep_mlp(state)

        x = x.unsqueeze(0)
        gru_out, hiddens = self.gru(x, hiddens)
        x = x + gru_out
        x = x.squeeze(0)

        return self.to_pred(x), hiddens

from torch.optim.lr_scheduler import CosineAnnealingLR

evo_strat = EvoStrategy(
    Model(),
    environment = HumanoidEnvironment(
        repeats = 1,
        render_every_eps = 200
    ),
    num_generations = 50_000,
    noise_population_size = 200,
    noise_low_rank = 1,
    noise_scale = 1e-2,
    noise_scale_clamp_range = (5e-3, 2e-2),
    learned_noise_scale = True,
    use_sigma_optimizer = True,
    learning_rate = 1e-3,
    noise_scale_learning_rate = 1e-4,
    use_scheduler = True,
    scheduler_klass = CosineAnnealingLR,
    scheduler_kwargs = dict(T_max = 50_000)
)

evo_strat()
