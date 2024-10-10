import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import filelock
import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch

from typing import Any, Dict, Optional, Tuple, Union



class PointmazeWrapper(gym.Wrapper):

    maze_to_start = {
        "PointMaze_Open-v3": np.array([2,1]),
        "PointMaze_UMaze-v3": np.array([3,1]),
        "PointMaze_Medium-v3": np.array([6,1]),
        "PointMaze_Large-v3": np.array([7,1])
    }

    maze_to_goal = {
        "PointMaze_Open-v3": np.array([1,1]),
        "PointMaze_UMaze-v3": np.array([1,1]),
        "PointMaze_Medium-v3": np.array([1,6]),
        "PointMaze_Large-v3": np.array([1,10])
    }

    def __init__(self,env) -> gym.Env:
        super().__init__(env)
        self.xy = torch.Tensor(env.unwrapped.maze.cell_rowcol_to_xy(PointmazeWrapper.maze_to_goal[env.spec.id]))

    def reset(self,seed:int=None,options:Dict[str,Any]=None) -> Tuple[Any,Dict[str,Any]]:
        #### fix goal and start position
        if options is None: options = {}
        return self.env.reset(
            seed=seed,
            options={
                "reset_cell": PointmazeWrapper.maze_to_start[self.spec.id],
                "goal_cell": PointmazeWrapper.maze_to_goal[self.spec.id],
                **options 
            }
        )
    
    def termination_fn(self,act:torch.Tensor,next_obs:torch.Tensor):
        pos = next_obs[:,:2]
        distance = torch.norm(self.xy - pos,dim=1)
        done = distance < 0.5
        done = done[:,None]
        return done
    