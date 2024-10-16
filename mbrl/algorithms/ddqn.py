import os.path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize_scalar
from torch.optim import Adam

from mbrl.models import ModelEnv
from mbrl.planning import Agent
from mbrl.third_party.pytorch_sac_pranz24.model import weights_init_

def compute_k(delta, x):
    return -np.mean(np.log(1 - delta * x))


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, negative_activation):
        super(QNetwork, self).__init__()

        self.negative_activation = negative_activation

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        if self.negative_activation:
            return -F.softplus(x)
        else:
            return x

class DDQNAgent(Agent):
    def __init__(self, num_inputs, action_space, strategy, num_epochs, args):
        self.args = args
        self.gamma = args.gamma
        self.tau = args.tau
        self.strategy = strategy
        self.num_epochs = num_epochs
        self.action_space = action_space

        self.target_update_interval = args.target_update_interval

        self.device = args.device

        self.q_network = QNetwork(num_inputs, action_space.n, args.hidden_size, args.negative_activation).to(self.device)
        self.optimizer = Adam(self.q_network.parameters(), lr=args.lr)

        self.target_network = QNetwork(num_inputs, action_space.n, args.hidden_size, args.negative_activation).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

    def compute_epsilon(self, epoch):
        end_epoch = int(self.strategy.exploration_fraction * self.num_epochs)
        if epoch >= end_epoch:
            return self.strategy.exploration_final_eps
        else:
            return self.strategy.exploration_initial_eps - (self.strategy.exploration_initial_eps - self.strategy.exploration_final_eps) * (epoch / end_epoch)

    def select_action(self, state, batched=False, evaluate=False, epoch=None, model_env: Optional[ModelEnv]=None):
        state = torch.FloatTensor(state)
        if not batched:
            state = state.unsqueeze(0)
        state = state.to(self.device)
        q_values = self.q_network(state)

        if self.strategy.name == "epsilon-greedy" and not evaluate and epoch is not None:
            epsilon = self.compute_epsilon(epoch)
            if batched:
                random_actions = torch.randint(0, q_values.shape[-1], (q_values.shape[0],))
                greedy_actions = torch.argmax(q_values, dim=-1)
                random_choice = torch.rand(q_values.shape[0]) < epsilon
                action = torch.where(random_choice, random_actions, greedy_actions)
            else:
                if np.random.rand() < epsilon:
                    action = torch.randint(0, q_values.shape[-1], (1,))
                else:
                    action = torch.argmax(q_values, dim=-1)
        elif self.strategy.name == "count-based" and not evaluate:
            actions = torch.arange(self.action_space.n).repeat(state.shape[0], 1)
            states = state.unsqueeze(1).repeat(1, self.action_space.n, 1)
            div, _, _ = model_env.dynamics_model.jensen_renyi_divergence(actions, {"obs": states})
            counts = 1 / (torch.exp(div + 1e-6) - 1)
            bonus = self.strategy.beta * torch.sqrt(1 / (counts + 0.01))
            action = torch.argmax(q_values + bonus, dim=-1)
        elif self.strategy.name == "imed" and not evaluate:
            actions = torch.arange(self.action_space.n).repeat(state.shape[0], 1)
            states = state.unsqueeze(1).repeat(1, self.action_space.n, 1)
            div, next_states, rewards = model_env.dynamics_model.jensen_renyi_divergence(actions, {"obs": states})
            terminated = model_env.termination_fn(actions.reshape(actions.shape[0] * actions.shape[1], -1),
                                     next_states.reshape(next_states.shape[0] * next_states.shape[1], -1))
            gamma = self.gamma * (~terminated.reshape(next_states.shape[0], next_states.shape[1]))


            counts = 1 / (torch.exp(div + 1e-6) - 1)
            # head, act, obs_dim
            v, _ = self.q_network(next_states).max(dim=-1)

            qs = rewards.squeeze() + gamma * v
            q_mean = qs.mean(axis=0)
            best_action = q_mean.argmax(axis=-1)
            mu = q_mean[best_action]
            best_actions = torch.arange(self.action_space.n)[torch.isclose(q_mean, mu, 1e-4)].cpu().numpy()

            deltas = (qs - mu).float().cpu().numpy()
            M = 0
            u = (1 / (M - mu) - 1e-6).item()
            ks = np.zeros(self.action_space.n)
            for action in range(self.action_space.n):
                if action not in best_actions:
                    res = minimize_scalar(compute_k, bounds=(0, u), args=(deltas[:, action],), method="bounded")
                    ks[action] = -res.fun

            hs = counts * ks + torch.log(counts)
            action = torch.argmin(hs, dim=-1, keepdim=True)
        else:
            action = torch.argmax(q_values, dim=-1)

        if batched:
            return action.detach().cpu().numpy()
        return action.detach().cpu().numpy()[0]

    def act(
        self, obs: np.ndarray, sample: bool = False, batched: bool = False, **_kwargs
    ) -> np.ndarray:
        with torch.no_grad():
            return self.select_action(
                obs, batched=batched, evaluate=not sample, **_kwargs
            )

    def update_parameters(
            self, memory, batch_size, updates, logger=None, reverse_mask=False
    ):
        (
            state_batch,
            action_batch,
            next_state_batch,
            reward_batch,
            mask_batch,
            _,
        ) = memory.sample(batch_size).astuple()

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.LongTensor(action_batch).to(self.device).unsqueeze(1)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        if reverse_mask:
            mask_batch = mask_batch.logical_not()

        with torch.no_grad():
            online_next_q_values = self.q_network(next_state_batch)
            _, next_actions = online_next_q_values.max(dim=1, keepdim=True)

            next_q_values = self.target_network(next_state_batch)
            next_q_value = next_q_values.gather(1, next_actions)

            td_target = reward_batch + mask_batch * self.gamma * next_q_value

        current_q_values = self.q_network(state_batch)
        current_q_value = current_q_values.gather(1, action_batch)

        loss = F.mse_loss(current_q_value, td_target)

        # with torch.no_grad():
        #     target_max, _ = self.target_network(next_state_batch).max(dim=1, keepdim=True)
        #     td_target = reward_batch + mask_batch * self.gamma * target_max
        #
        # old_val = self.q_network(state_batch).gather(1, action_batch)
        # loss = F.mse_loss(td_target, old_val)

        self.optimizer.zero_grad()
        loss.backward()

        # torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.args.grad_clip)
        self.optimizer.step()

        if updates % self.target_update_interval == 0:
            for target_network_param, q_network_param in zip(self.target_network.parameters(), self.q_network.parameters()):
                target_network_param.data.copy_(
                    self.tau * q_network_param.data + (1.0 - self.tau) * target_network_param.data
                )

        if logger is not None:
            logger.log("train/batch_reward", reward_batch.mean(), updates)
            logger.log("train_q_network/loss", loss, updates)

        return (loss.item(),)

    def save_checkpoint(self, env_name=None, suffix="", ckpt_path=None):
        if ckpt_path is None:
            assert env_name is not None
            if not os.path.exists("checkpoints/"):
                os.mkdir("checkpoints/")
            ckpt_path = "checkpoints/ddqn_{}_{}".format(env_name, suffix)
        print("Saving model to {}".format(ckpt_path))

        torch.save(
            {
                "q_network_state_dict": self.q_network.state_dict(),
                "target_network_state_dict": self.target_network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            ckpt_path,
        )

    def load_checkpoint(self, ckpt_path, evaluate=False):
        print("Loading model from {}".format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
            self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            if evaluate:
                self.q_network.eval()
                self.target_network.eval()
            else:
                self.q_network.train()
                self.target_network.train()






