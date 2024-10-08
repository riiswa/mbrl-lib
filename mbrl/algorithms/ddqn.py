import os.path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from mbrl.planning import Agent
from mbrl.third_party.pytorch_sac_pranz24.model import weights_init_


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
    def __init__(self, num_inputs, action_space, args):
        self.args = args
        self.gamma = args.gamma
        self.tau = args.tau

        self.target_update_interval = args.target_update_interval

        self.device = args.device

        self.q_network = QNetwork(num_inputs, action_space.n, args.hidden_size, args.negative_activation).to(self.device)
        self.optimizer = Adam(self.q_network.parameters(), lr=args.lr)

        self.target_network = QNetwork(num_inputs, action_space.n, args.hidden_size, args.negative_activation).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

    def select_action(self, state, batched=False, evaluate=False):
        state = torch.FloatTensor(state)
        if not batched:
            state = state.unsqueeze(0)
        state = state.to(self.device)
        q_values = self.q_network(state)
        action = torch.argmax(q_values, dim=-1)
        if batched:
            return action.detach().cpu().numpy()
        return action.detach().cpu().numpy()[0]

    def act(
        self, obs: np.ndarray, sample: bool = False, batched: bool = False, **_kwargs
    ) -> np.ndarray:
        with torch.no_grad():
            return self.select_action(
                obs, batched=batched, evaluate=not sample
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
        #     target_max, _ = self.target_network(next_state_batch).max(dim=1)
        #     td_target = reward_batch + mask_batch * self.gamma * target_max
        # old_val = self.q_network(state_batch).gather(1, action_batch).squeeze()
        # loss = F.mse_loss(td_target, old_val)

        self.optimizer.zero_grad()
        loss.backward()
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






