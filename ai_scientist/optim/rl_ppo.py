"""Minimal PPO implementation for Stellarator Design Optimization (StellarForge Phase 3).

This module provides a lightweight PPO (Proximal Policy Optimization) agent
tailored for the continuous control task of refining stellarator boundary coefficients.
It is designed to work with the `StellaratorEnv`.
"""

import logging
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

_LOGGER = logging.getLogger(__name__)


def layer_init(
    layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0
) -> nn.Module:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, observation_shape: int, action_shape: int, hidden_dim: int = 64):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(observation_shape, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(observation_shape, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, action_shape), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_shape))

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x)

    def get_action_and_value(
        self, x: torch.Tensor, action: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        # Clamp logstd to prevent exp underflow (-5 → ~0.0067) and explosion (+2 → ~7.4)
        # Note: min=-5.0 allows reasonable exploration; min=-20.0 would collapse exploration
        action_logstd = torch.clamp(action_logstd, min=-5.0, max=2.0)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()

        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(x),
        )


class PPOBuffer:
    """Simple replay buffer for PPO."""

    def __init__(self, obs_dim, act_dim, size, device="cpu"):
        self.obs = torch.zeros((size, obs_dim), dtype=torch.float32).to(device)
        self.actions = torch.zeros((size, act_dim), dtype=torch.float32).to(device)
        self.logprobs = torch.zeros(size, dtype=torch.float32).to(device)
        self.rewards = torch.zeros(size, dtype=torch.float32).to(device)
        self.dones = torch.zeros(size, dtype=torch.float32).to(device)
        self.values = torch.zeros(size, dtype=torch.float32).to(device)
        self.ptr = 0
        self.size = size
        self.device = device

    def add(self, obs, action, logprob, reward, done, value):
        if self.ptr >= self.size:
            _LOGGER.warning(
                "PPOBuffer overflow: buffer full (size=%d), dropping experience. "
                "Consider increasing buffer size or calling reset() more frequently.",
                self.size,
            )
            return
        self.obs[self.ptr] = torch.tensor(obs).to(self.device)
        self.actions[self.ptr] = action
        self.logprobs[self.ptr] = logprob
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        self.ptr += 1

    def get(self):
        return (
            self.obs[: self.ptr],
            self.actions[: self.ptr],
            self.logprobs[: self.ptr],
            self.rewards[: self.ptr],
            self.dones[: self.ptr],
            self.values[: self.ptr],
        )

    def reset(self):
        self.ptr = 0


class PPOEngine:
    """Trainer for the PPO Agent."""

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.2,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: float | None = None,
        device: str = "cpu",
    ):
        self.agent = Agent(input_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=lr, eps=1e-5)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.device = device

    def train_step(
        self, buffer: PPOBuffer, next_value: torch.Tensor, next_done: torch.Tensor
    ):
        obs, actions, logprobs, rewards, dones, values = buffer.get()

        # Bootstrap value if not done
        with torch.no_grad():
            advantages = torch.zeros_like(rewards).to(self.device)
            lastgaelam = 0
            for t in reversed(range(buffer.ptr)):
                if t == buffer.ptr - 1:
                    nextnonterminal = 1.0 - next_done.float()
                    nextvalues = next_value.squeeze()
                else:
                    nextnonterminal = 1.0 - dones[t].float()
                    nextvalues = values[t + 1]

                delta = (
                    rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                )

            returns = advantages + values

        # Optimize Policy and Value
        # Flatten batch (we only have one batch here effectively)
        b_obs = obs
        b_logprobs = logprobs
        b_actions = actions
        b_advantages = advantages
        b_returns = returns

        # Minibatch updates could be done here, but for simplicity we do full batch update
        # since batch size is small (e.g. 50-100 steps per refinement cycle).

        new_action, new_logprob, entropy, new_value = self.agent.get_action_and_value(
            b_obs, b_actions
        )
        logratio = new_logprob - b_logprobs
        ratio = logratio.exp()

        with torch.no_grad():
            # calculate approx_kl http://joschu.net/blog/kl-approx.html
            approx_kl = ((ratio - 1) - logratio).mean()

        mb_advantages = b_advantages
        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
            mb_advantages.std() + 1e-8
        )

        # Policy loss
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * torch.clamp(
            ratio, 1 - self.clip_coef, 1 + self.clip_coef
        )
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Value loss
        newvalue = new_value.view(-1)
        v_loss = 0.5 * ((newvalue - b_returns) ** 2).mean()

        entropy_loss = entropy.mean()
        loss = pg_loss - self.ent_coef * entropy_loss + self.vf_coef * v_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return loss.item(), approx_kl.item()
