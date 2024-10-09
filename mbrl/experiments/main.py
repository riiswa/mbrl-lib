import hydra
import math
import numpy as np
import omegaconf
import torch

import gymnasium as gym

from mbrl.algorithms import mbpo_discrete


def make_env(env_name):
    if env_name == "CartPole-v1":
        env = gym.make("CartPole-v1", max_episode_steps=200)
        def termination_fn(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
            assert len(next_obs.shape) == 2

            x, theta = next_obs[:, 0], next_obs[:, 2]

            x_threshold = 2.4
            theta_threshold_radians = 12 * 2 * math.pi / 360
            not_done = (
                    (x > -x_threshold)
                    * (x < x_threshold)
                    * (theta > -theta_threshold_radians)
                    * (theta < theta_threshold_radians)
            )
            done = ~not_done
            done = done[:, None]
            return done
        return env, termination_fn
    if env_name == "MountainCar-v0":
        env = gym.make("MountainCar-v0", max_episode_steps=200)

        def termination_fn(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
            assert len(next_obs.shape) == 2

            x = next_obs[:, 0]  # position of the car

            goal_position = 0.5
            return (x >= goal_position)[:, None]
        return env, termination_fn
    else:
        raise NotImplementedError

@hydra.main(config_path="conf", config_name="main")
def run(cfg: omegaconf.DictConfig):
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    env, term_fn = make_env(cfg.overrides.env)

    return mbpo_discrete.train(env, None, term_fn, cfg)

if __name__ == "__main__":
    run()