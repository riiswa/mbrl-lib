import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import filelock
import gymnasium as gym
import minari
import numpy as np

from typing import Any, Dict, Optional, Tuple, Union



class Pointmaze(gym.Env):

    """
    Pointmaze environment based on these Gymnasium environents: PointMaze_UMaze-v3, PointMaze_Medium-v3, PointMaze_Large-v3.
    See: https://robotics.farama.org/envs/maze/point_maze/.

    The available mazes are: umaze, medium, large.

    The rough observation space is a dictionary with the following keys:
    - observation: `Box(-inf, inf, (4,), float64)
    - desired_goal: `Box(-inf, inf, (2,), float64)
    - achieved_goal: `Box(-inf, inf, (2,), float64)
    
    The rough action space is `Box(-1.0, 1.0, (2,), float32)`.

    When using batched environments, autoreset is always set to True.

    Args:
        maze (str): Maze to use. Default is "umaze".
        autoreset (bool): Automatically reset the environment. Default is True.
        max_episode_steps (int): Maximum number of steps per episode. Default is None.
        render_mode (str): Render mode. Default is None.
        batched (bool): Whether to use a batch of environments. Default is True.
        batch_size (int): Number of environments in the batch. Default is 1.
        asynchrone (bool): Whether to use asynchroneous vectorized environments. Default is False.
    """

    mazes = ["umaze","medium","large"]

    def __init__(
        self,
        maze:str="umaze",
        n_actions:float=3,
        std:float=0.1,
        autoreset: bool = True,
        max_episode_steps: Optional[int] = None,
        render_mode: Optional[str] = None,
        batched: bool = False,
        batch_size:int = 1,
        asynchrone:bool = False) -> gym.Env:

        assert maze in Pointmaze.mazes, f"Invalid maze: {maze}. Available mazes: {Pointmaze.mazes}."
        assert (0 < std) and (2*std < 1), f"Invalid std: {std}. Must be in (0,1)."
        self.n_actions = n_actions
        self.std = std
        self.min_action, self.max_action = -1 + 2*std, 1 - 2*std
        self.actions = np.linspace(self.min_action,self.max_action,n_actions) 
        self.batched = batched
        self.batch_size = batch_size
        self.asynchrone = asynchrone

        dataset = minari.load_dataset(f"pointmaze-{maze}-v2",download=True)
        dataset.set_seed(0)
        make_env_fn = lambda: dataset.recover_environment(
            eval_env=False,
            autoreset=autoreset,
            max_episode_steps=max_episode_steps,
            render_mode=render_mode
        )

        lock = filelock.FileLock("/tmp/point_maze.xml.lock")
        with lock:
            # create a batch of environments
            if self.batched:
                if self.asynchrone: self._env = gym.vector.AsyncVectorEnv([make_env_fn for _ in range(batch_size)])
                else: self._env = gym.vector.SyncVectorEnv([make_env_fn for _ in range(batch_size)])
            # create a single environment
            else: self._env = make_env_fn()

        # attributes
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space
        self.metadata = self._env.metadata
        self.render_mode = self._env.render_mode
        self.reward_range = self._env.reward_range
        self.spec = self._env.spec
        
        # modifiers
        self._obs_modifier = lambda obs: obs
        self._action_modifier = lambda action: self._discretize_action(action)

    def _discretize_action(self,action:np.ndarray) -> np.ndarray:
        if self.batched:
            return np.array([[self.actions[np.argmin(np.abs(a - self.actions))] for a in act] for act in action])
        else:
            return np.array([self.actions[np.argmin(np.abs(a - self.actions))] for a in action])

    def reset(self,seed:int=None,options:Dict[str,Any]=None) -> Tuple[Any,Dict[str,Any]]:
        obs, infos = self._env.reset(seed=seed,options=options)
        obs["obs"] = self._obs_modifier(obs["observation"])
        obs["goal"] = obs["desired_goal"]
        obs["pos"] = obs["achieved_goal"]
        state = np.concatenate([obs["obs"],obs["goal"],obs["pos"]],axis=-1)
        return state, infos
    
    def step(self,action:np.ndarray) -> Tuple[Any,Union[float,np.ndarray],Union[bool,np.ndarray],Union[bool,np.ndarray],Dict[str,Any]]:
        obs, reward, done, truncated, infos = self._env.step(self._action_modifier(action))
        obs["obs"] = self._obs_modifier(obs["observation"])
        obs["goal"] = obs["desired_goal"]
        obs["pos"] = obs["achieved_goal"]
        # concatanate all obs into a single array state
        state = np.concatenate([obs["obs"],obs["goal"],obs["pos"]],axis=-1)
        done = reward > 0
        return state, reward, done, truncated, infos
    
    def render(self) -> None:
        self._env.render()

    def close(self) -> None:
        self._env.close()
    
    @property
    def unwrapped(self):
        return self._env.unwrapped
    
    @property
    def np_random(self):
        return self._env.np_random
    
    def compute_reward(self,achieved_goal:np.ndarray,desired_goal:np.ndarray,info:Dict[str,Any]) -> float:
        return self._env.compute_reward(achieved_goal,desired_goal,info)
