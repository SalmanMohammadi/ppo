import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from jaxtyping import Float, Int64, Array
from gymnasium.spaces import Box, Discrete
from torch.distributions.normal import Normal
from typing import Union
from torch.distributions.categorical import Categorical
import gymnasium as gym
from typing import Union
import scipy
import torch.optim as optim
import logging
from yaml import safe_load
import numpy as np
import sys
import plotext as plt


def make_layers(n_layers: int, hidden_dim: int, activation=nn.Tanh):
    return [l for _ in range(n_layers) for l in (nn.Linear(hidden_dim, hidden_dim), activation())]


def discounted_cumsum(x: np.ndarray, gamma: float) -> np.ndarray:
    """
    rllab method for exponentially discounted cumulative sum of vectors.
    Args:
        x: A vector of length n [x0, x1, ..., xn]
        gamma: discount factor in [0, 1]
    """
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


class Actor(nn.Module):
    def log_prob(
        self, pi: Union[Categorical, Normal], actions: Float[Tensor, "batch act_dim"]
    ) -> Float[Tensor, "batch act_dim"]:
        raise NotImplementedError

    def forward(
        self, obs: Float[Tensor, "batch obs_dim"], actions: Float[Tensor, "batch act_dim"] = None  # pyright: ignore
    ) -> Union[Categorical, tuple[Categorical, Float[Tensor, "batch act_dim"]]]:
        pi = self.distribution(obs)
        if actions is not None:
            return pi, self.log_prob(pi, actions)
        return pi


class CategoricalActor(Actor):
    def __init__(self, obs_dim: int, act_dim: Int64, hidden_dim: int = 64, n_layers: int = 4, activation=nn.Tanh):
        super().__init__()
        self.theta = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            activation(),
            *make_layers(n_layers, hidden_dim),
            nn.Linear(hidden_dim, act_dim),
        )

    def distribution(self, obs: Float[Tensor, "batch obs_dim"]) -> Categorical:
        return Categorical(logits=self.theta(obs))

    def log_prob(self, pi: Categorical, actions: Float[Tensor, "batch act_dim"]) -> Float[Tensor, "batch act_dim"]:
        return pi.log_prob(actions)


class GaussianActor(Actor):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 64, n_layers: int = 4, activation=nn.Tanh):
        super().__init__()
        # why don't we also learn a scale for uniform variance?
        self.log_sigma = nn.Parameter(-0.5 * torch.ones(act_dim))
        self.mu = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            activation(),
            *make_layers(n_layers, hidden_dim),
            nn.Linear(hidden_dim, act_dim),
        )

    def distribution(self, obs: Float[Tensor, "batch obs_dim"]) -> Normal:
        return Normal(self.mu(obs), self.log_sigma.exp())

    def log_prob(self, pi: Normal, actions: Float[Tensor, "batch act_dim"]) -> Float[Tensor, "batch act_dim"]:
        return pi.log_prob(actions).sum(axis=-1)


class Critic(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int = 64, n_layers: int = 4, activation=nn.Tanh):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), activation(), *make_layers(n_layers, hidden_dim), nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs: Float[Tensor, "batch obs_dim"]) -> Float[Tensor, "batch"]:
        return F.tanh(self.phi(obs)).squeeze()


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_space: Box,
        act_space: Union[Box, Discrete],
        hidden_dim: int = 64,
        n_layers: int = 4,
        activation=nn.Tanh,
    ):
        super().__init__()
        assert isinstance(act_space, Box) or isinstance(act_space, Discrete)
        assert isinstance(obs_space, Box)
        if isinstance(act_space, Box):
            self.pi = GaussianActor(
                obs_space.shape[0], act_space.shape[0], hidden_dim=hidden_dim, n_layers=n_layers, activation=activation
            )
        elif isinstance(act_space, Discrete):
            self.pi = CategoricalActor(
                obs_space.shape[0], act_space.n, hidden_dim=hidden_dim, n_layers=n_layers, activation=activation
            )
        self.v = Critic(obs_space.shape[0], hidden_dim=hidden_dim, n_layers=n_layers, activation=activation)

    def step(
        self, obs: Float[Tensor, "batch obs_dim"]
    ) -> tuple[Float[Tensor, "batch act_dim"], Float[Tensor, "batch"], Float[Tensor, "batch act_dim"]]:
        with torch.no_grad():
            pi = self.pi(obs)
            a = pi.sample()
            logp_a = self.pi.log_prob(pi, a)  # pyright: ignore
            v = self.v(obs)
        return a, v, logp_a

    def act(self, obs: Float[Tensor, "batch obs_dim"]) -> Float[Tensor, "batch act_dim"]:
        return self.pi(obs).sample()


class PPO:
    def __init__(
        self,
        env: gym.Env,
        ac: ActorCritic,
        gamma: float,
        lmbda: float,
        eps: float,
        steps_per_epoch: int,
        episode_length: int,
        device: torch.device = torch.device("cpu"),
    ):
        self.env = env
        self.ac = ac
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps = eps
        self.episode_length = episode_length
        self.steps_per_epoch = steps_per_epoch
        self.reset_buffers()
        self.device = device

    def store_values(
        self,
        obs: Float[Tensor, "batch obs_dim"],
        a: float,
        logp_a: Float[Tensor, "batch act_dim"],
        r: float,
        v: float,
    ):
        self.obs_buffer[self.idx] = obs
        self.act_buffer[self.idx] = a
        self.logp_a_buffer[self.idx] = logp_a
        self.r_buffer[self.idx] = r
        self.v_buffer[self.idx] = v
        self.idx += 1

    def buffers_to_tensors(self):
        self.obs_buffer = torch.as_tensor(self.obs_buffer, dtype=torch.float32, device=self.device)
        self.act_buffer = torch.as_tensor(self.act_buffer, dtype=torch.float32, device=self.device)
        self.logp_a_buffer = torch.as_tensor(self.logp_a_buffer, dtype=torch.float32, device=self.device)
        self.advantage_buffer = torch.as_tensor(self.advantage_buffer, dtype=torch.float32, device=self.device)
        self.return_buffer = torch.as_tensor(self.return_buffer, dtype=torch.float32, device=self.device)
        self.r_buffer = torch.as_tensor(self.r_buffer, dtype=torch.float32, device=self.device)
        self.v_buffer = torch.as_tensor(self.v_buffer, dtype=torch.float32, device=self.device)

    def reset_buffers(self):
        self.obs_buffer = np.zeros(
            (
                self.steps_per_epoch,
                *self.env.observation_space.shape,
            )
        )
        self.act_buffer = np.zeros((self.steps_per_epoch))
        self.logp_a_buffer = np.zeros((self.steps_per_epoch))
        self.r_buffer = np.zeros((self.steps_per_epoch))
        self.v_buffer = np.zeros((self.steps_per_epoch))
        self.advantage_buffer = np.zeros((self.steps_per_epoch))
        self.return_buffer = np.zeros((self.steps_per_epoch))
        self.idx = 0
        self.advantage_start_idx = 0

    def estimate_advantage_and_return(self, terminal_value=0):
        """
        Estimates the advantage for the current trajectory using Generalized Advantage Estimation (https://arxiv.org/pdf/1506.02438.pdf).
        (one step advantage estimation)
        """
        advantage_slice = slice(self.advantage_start_idx, self.idx)
        r_buffer = np.append(self.r_buffer[advantage_slice], terminal_value)
        v_buffer = np.append(self.v_buffer[advantage_slice], terminal_value)
        deltas = r_buffer[:-1] + self.gamma + v_buffer[1:] - v_buffer[:-1]
        self.advantage_buffer[advantage_slice] = discounted_cumsum(deltas, self.gamma * self.lmbda)
        self.return_buffer[advantage_slice] = discounted_cumsum(r_buffer, self.gamma)[:-1]
        self.advantage_start_idx = self.idx

    def policy_loss(
        self, adv: Float[Tensor, "buffer_size"]
    ) -> tuple[Float[Tensor, "1"], Float[Tensor, "1"], Float[Tensor, "1"]]:
        _, logp_a = self.ac.pi(self.obs_buffer, self.act_buffer)
        ratio = (logp_a - self.logp_a_buffer).exp()  # exp(log(a) - log(b)) = a/b
        clipped_ratio = torch.clamp(ratio, 1 - self.eps, 1 + self.eps)  # PPO clip step
        loss = torch.min(ratio * adv, clipped_ratio * adv)  # PPO pessimistic update step
        return (
            -loss.mean(),
            (0.5 * (logp_a - self.logp_a_buffer).pow(2)).mean(),
            (ratio - clipped_ratio).mean() / ratio.mean(),
        )

    def v_loss(self):
        ret = (self.return_buffer - self.return_buffer.mean()) / self.return_buffer.std()
        return ((self.ac.v(self.obs_buffer) - ret).pow(2)).mean()

    def policy_update(self, opt: optim.Optimizer, train_steps: int):
        total_loss, total_kl, ratios = 0, 0, 0
        adv = self.advantage_buffer
        adv = (adv - adv.mean()) / adv.std()
        for i in range(train_steps):
            opt.zero_grad()
            loss, kl, ratio = self.policy_loss(adv)
            total_loss += loss
            total_kl += kl
            ratios += ratio
            # if kl > 0.05:
            #     break
            loss.backward()
            opt.step()
        return total_loss / train_steps, total_kl / train_steps, ratios / train_steps

    def v_update(self, opt: optim.Optimizer, train_steps: int):
        total_mse = 0
        for i in range(train_steps):
            opt.zero_grad()
            loss = self.v_loss()
            total_mse += loss
            loss.backward()
            opt.step()
        return total_mse / train_steps

    def train(
        self,
        pi_opt: optim.Optimizer,
        v_opt: optim.Optimizer,
        epochs: int,
        pi_train_steps: int,
        v_train_steps: int,
    ):
        # epoch_returns = []
        # pi_losses = []
        # kls = []
        # v_losses = []
        # ratios = []
        obs, *_ = self.env.reset()
        ep_len = 0
        ep_r = 0
        for epoch in range(epochs):
            epoch_return = 0
            n_ep = 0
            for t in range(self.steps_per_epoch):
                a, v, logp_a = self.ac.step(torch.as_tensor(obs, dtype=torch.float32, device=self.device))
                a = a.cpu().item()
                obs_, reward, terminated, *_ = env.step(a)

                ep_r += reward
                ep_len += 1

                self.store_values(obs, a, logp_a, reward, v.cpu())  # pyright: ignore
                obs = obs_
                epoch_ended = t == self.steps_per_epoch - 1
                terminated = terminated or (ep_len == self.episode_length)
                if terminated or epoch_ended:
                    if epoch_ended and not terminated:
                        logging.debug(f"Trajectory for epoch {epoch} ended early at {ep_len} steps")
                    if terminated:
                        logging.debug(f"Trajectory ended with {ep_len} steps and {ep_r} return")
                    if epoch_ended or (ep_len == self.episode_length):
                        _, v, _ = self.ac.step(torch.as_tensor(obs, dtype=torch.float32, device=self.device))
                        v = v.cpu()
                    else:
                        v = 0
                    self.estimate_advantage_and_return(v)
                    obs, *_ = env.reset()
                    logging.info(f"Episode length: {ep_len} return {ep_r}")
                    ep_len = 0
                    # epoch_return += ep_r
                    n_ep += 1
                    # epoch_returns.append(ep_r)
                    ep_r = 0

            self.buffers_to_tensors()
            pi_loss, kl, ratio = self.policy_update(pi_opt, pi_train_steps)
            # pi_losses.append(pi_loss.detach())
            # kls.append(kl.detach())
            # ratios.append((ratio * 100).detach())
            v_loss = self.v_update(v_opt, v_train_steps)
            # v_losses.append(v_loss.detach())
            self.reset_buffers()


if __name__ == "__main__":
    with open(sys.argv[1], "r") as f:
        config = safe_load(f)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logging.info(config)
    device = torch.device(config["device"])

    env = gym.make(**config["env"])
    ac = ActorCritic(env.observation_space, env.action_space, **config["ac"], activation=nn.ReLU)  # pyright: ignore
    ac.to(device)
    ac.train()
    ppo = PPO(env, ac, **config["ppo"], device=device)
    pi_opt = optim.Adam(ac.pi.parameters(), lr=config["pi_lr"])
    v_opt = optim.Adam(ac.v.parameters(), lr=config["v_lr"])
    ppo.train(pi_opt, v_opt, **config["train"])
    env.close()

    env = gym.make(config["env"]["id"], render_mode="human")
    obs, *_ = env.reset()
    ac.eval()
    with torch.no_grad():
        ret = 0
        t = 0
        for _ in range(10000):
            a, *_ = ac.step(torch.as_tensor(obs, dtype=torch.float32))
            obs, reward, terminated, *_ = env.step(a.item())
            ret += reward
            t += 1
            if terminated:
                logging.info(f"Finished with {t} steps and {ret} return")
                ret = 0
                t = 0
                obs, *_ = env.reset()
    env.close()
