import hydra
import math
import numpy as np
import omegaconf
import torch

import gymnasium as gym
import gymnasium.wrappers as wrappers
import gymnasium_robotics

from mbrl.algorithms import mbpo_discrete
from mbrl.env.pointmaze_discrete import PointmazeWrapper
from mbrl.env.antmaze_discrete import AntMazeWrapper


def make_env(env_name,):

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
        return env, 

    if "pointmaze_D" in env_name:
        n_actions = int(env_name.split("_")[1][1:])
        maze = env_name.split("_")[2][1:]
        std = float(env_name.split("_")[3][1:])/100
        print(f"n_actions: {n_actions}, maze: {maze}, std: {std}")
        maze_to_name = {
            "open": "PointMaze_Open-v3",
            "umaze": "PointMaze_UMaze-v3",
            "medium": "PointMaze_Medium-v3",
            "large": "PointMaze_Large-v3"
        }
        env = gym.make(
            maze_to_name[maze],
            max_episode_steps=500,
            render_mode="human",
            continuing_task=False
        )

        env = PointmazeWrapper(env)

        termination_fn = env.termination_fn
        
        min_action, max_action = -1 + 2 * std, 1 - 2 * std
        available_actions_range = np.linspace(min_action,max_action,n_actions)

        available_actions = np.array([[a1,a2] for a1 in available_actions_range for a2 in available_actions_range])

        env = wrappers.TransformObservation(
            env, 
            lambda o: np.concatenate([o["observation"], o["desired_goal"] - o["achieved_goal"]]), 
            gym.spaces.Box(-np.inf, np.inf, (6,))
        )

        env = wrappers.TransformAction(env, lambda a: available_actions[a], gym.spaces.Discrete(n_actions**2))
        env = wrappers.TransformReward(env, lambda r: r - 1)
    
        return env, termination_fn

    if "antmaze_D" in env_name:
        n_actions = int(env_name.split("_")[1][1:])
        maze = env_name.split("_")[2][1:]
        std = float(env_name.split("_")[3][1:])/100
        print(f"n_actions: {n_actions}, maze: {maze}, std: {std}")
        maze_to_name = {
            "open": "AntMaze_Open-v4",
            "umaze": "AntMaze_UMaze-v4",
            "medium": "AntMaze_Medium-v4",
            "large": "AntMaze_Large-v4"
        }
        env = gym.make(
            maze_to_name[maze],
            max_episode_steps=200,
            render_mode="human",
            continuing_task=False
        )

        env = AntMazeWrapper(env)

        termination_fn = env.termination_fn
        
        min_action, max_action = -1 + 2 * std, 1 - 2 * std
        available_actions_range = np.linspace(min_action,max_action,n_actions)

        available_actions = np.array([[a1,a2] for a1 in available_actions_range for a2 in available_actions_range])

        env = wrappers.TransformAction(env, lambda a: available_actions[a], gym.spaces.Discrete(n_actions**2))
        env = wrappers.TransformObservation(env, lambda o: o["observation"], gym.spaces.Box(-np.inf, np.inf, (27,)))
    
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