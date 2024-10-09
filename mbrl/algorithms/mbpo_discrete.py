# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
from typing import Optional, Sequence, cast

import gymnasium as gym
import hydra.utils
import numpy as np
import omegaconf
import torch
from omegaconf import open_dict
from torch.utils.tensorboard import SummaryWriter

import mbrl.constants
import mbrl.models
import mbrl.planning
import mbrl.types
import mbrl.util
import mbrl.util.common
import mbrl.util.math
from mbrl.algorithms.ddqn import DDQNAgent
from mbrl.third_party.pytorch_sac import VideoRecorder

MBPO_LOG_FORMAT = [
    ("epoch", "E", "int"),
    ("env_step", "S", "int"),
    ("episodic_reward", "R", "float"),
    ("episodic_length", "L", "int"),

]

def rollout_model_and_populate_sac_buffer(
    model_env: mbrl.models.ModelEnv,
    replay_buffer: mbrl.util.ReplayBuffer,
    agent: DDQNAgent,
    sac_buffer: mbrl.util.ReplayBuffer,
    rollout_horizon: int,
    batch_size: int,
    posterior_sampling: bool = False,
):
    batch = replay_buffer.sample(batch_size)
    initial_obs, *_ = cast(mbrl.types.TransitionBatch, batch).astuple()
    model_state = model_env.reset(
        initial_obs_batch=cast(np.ndarray, initial_obs),
        return_as_np=True,
    )
    accum_dones = np.zeros(initial_obs.shape[0], dtype=bool)
    obs = initial_obs
    for i in range(rollout_horizon):
        action = agent.act(obs, sample=False, batched=True, model_env=model_env)

        elite_models = model_env.dynamics_model.model.elite_models
        if posterior_sampling:
            model_env.dynamics_model.model.set_elite([np.random.choice(elite_models)])
        pred_next_obs, pred_rewards, pred_dones, model_state = model_env.step(
            action, model_state, sample=True
        )
        model_env.dynamics_model.model.set_elite(elite_models)

        truncateds = np.zeros_like(pred_dones, dtype=bool)
        sac_buffer.add_batch(
            obs[~accum_dones],
            action[~accum_dones],
            pred_next_obs[~accum_dones],
            pred_rewards[~accum_dones, 0],
            pred_dones[~accum_dones, 0],
            truncateds[~accum_dones, 0],
        )
        obs = pred_next_obs
        accum_dones |= pred_dones.squeeze()


def maybe_replace_sac_buffer(
    sac_buffer: Optional[mbrl.util.ReplayBuffer],
    obs_shape: Sequence[int],
    act_shape: Sequence[int],
    new_capacity: int,
    seed: int,
) -> mbrl.util.ReplayBuffer:
    if sac_buffer is None or new_capacity != sac_buffer.capacity:
        if sac_buffer is None:
            rng = np.random.default_rng(seed=seed)
        else:
            rng = sac_buffer.rng
        new_buffer = mbrl.util.ReplayBuffer(new_capacity, obs_shape, act_shape, rng=rng)
        if sac_buffer is None:
            return new_buffer
        (
            obs,
            action,
            next_obs,
            reward,
            terminated,
            truncated,
        ) = sac_buffer.get_all().astuple()
        new_buffer.add_batch(obs, action, next_obs, reward, terminated, truncated)
        return new_buffer
    return sac_buffer


def train(
    env: gym.Env,
    test_env: gym.Env,
    termination_fn: mbrl.types.TermFnType,
    cfg: omegaconf.DictConfig,
    silent: bool = False,
    work_dir: Optional[str] = None,
) -> np.float32:
    # ------------------- Initialization -------------------
    debug_mode = cfg.get("debug_mode", False)

    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    mbrl.planning.complete_agent_cfg(env, cfg.algorithm.agent)
    with open_dict(cfg.algorithm.agent):
        cfg.algorithm.agent.strategy = cfg.strategy
        cfg.algorithm.agent.num_epochs = cfg.overrides.num_epochs

    agent = cast(DDQNAgent, hydra.utils.instantiate(cfg.algorithm.agent))

    work_dir = work_dir or os.getcwd()
    # enable_back_compatible to use pytorch_sac agent
    logger = mbrl.util.Logger(work_dir, enable_back_compatible=True)
    logger.register_group(
        mbrl.constants.RESULTS_LOG_NAME,
        MBPO_LOG_FORMAT,
        color="green",
        dump_frequency=1,
    )
    writer = SummaryWriter(log_dir=work_dir)
    save_video = cfg.get("save_video", False)
    video_recorder = VideoRecorder(work_dir if save_video else None)

    rng = np.random.default_rng(seed=cfg.seed)
    torch_generator = torch.Generator(device=cfg.device)
    if cfg.seed is not None:
        torch_generator.manual_seed(cfg.seed)

    # -------------- Create initial overrides. dataset --------------
    dynamics_model = mbrl.util.common.create_one_dim_tr_model(cfg, obs_shape, (env.action_space.n.item(),), act_is_discrete=True)
    use_double_dtype = cfg.algorithm.get("normalize_double_precision", False)
    dtype = np.double if use_double_dtype else np.float32
    replay_buffer = mbrl.util.common.create_replay_buffer(
        cfg,
        obs_shape,
        act_shape,
        rng=rng,
        obs_type=dtype,
        action_type=dtype,
        reward_type=dtype,
    )
    random_explore = cfg.algorithm.random_initial_explore
    mbrl.util.common.rollout_agent_trajectories(
        env,
        cfg.algorithm.initial_exploration_steps,
        mbrl.planning.RandomAgent(env) if random_explore else agent,
        {} if random_explore else {"sample": True, "batched": False},
        replay_buffer=replay_buffer,
    )

    # ---------------------------------------------------------
    # --------------------- Training Loop ---------------------
    rollout_batch_size = (
        cfg.overrides.effective_model_rollouts_per_step * cfg.algorithm.freq_train_model
    )
    trains_per_epoch = int(
        np.ceil(cfg.overrides.epoch_length / cfg.overrides.freq_train_model)
    )
    updates_made = 0
    env_steps = 0
    model_env = mbrl.models.ModelEnv(
        env, dynamics_model, termination_fn, None, generator=torch_generator
    )
    model_trainer = mbrl.models.ModelTrainer(
        dynamics_model,
        optim_lr=cfg.overrides.model_lr,
        weight_decay=cfg.overrides.model_wd,
        logger=None if silent else logger,
    )
    epoch = 0
    sac_buffer = None
    obs, _ = env.reset()
    terminated = False
    truncated = False
    episodic_reward = 0
    episodic_length = 0
    while epoch < cfg.overrides.num_epochs:
        rollout_length = int(
            mbrl.util.math.truncated_linear(
                *(cfg.overrides.rollout_schedule + [epoch + 1])
            )
        )
        sac_buffer_capacity = rollout_length * rollout_batch_size * trains_per_epoch

        sac_buffer_capacity *= cfg.overrides.num_epochs_to_retain_sac_buffer
        sac_buffer = maybe_replace_sac_buffer(
            sac_buffer, obs_shape, act_shape, sac_buffer_capacity, cfg.seed
        )
        if terminated or truncated:
            logger.log_data(
                mbrl.constants.RESULTS_LOG_NAME,
                {
                    "epoch": epoch,
                    "env_step": env_steps,
                    "episodic_reward": episodic_reward,
                    "episodic_length": episodic_length,
                },
            )

            writer.add_scalar("episodic_reward", episodic_reward, epoch)
            writer.add_scalar("episodic_length", episodic_length, epoch)
            episodic_reward = 0
            episodic_length = 0
            epoch += 1
            obs, _ = env.reset()
        (
            next_obs,
            reward,
            terminated,
            truncated,
            _,
        ) = mbrl.util.common.step_env_and_add_to_buffer(
            env, obs, agent, {"epoch": epoch, "model_env": model_env, "sample": True}, replay_buffer
        )

        episodic_reward += reward
        episodic_length += 1

        # --------------- Model Training -----------------
        if (env_steps + 1) % cfg.overrides.freq_train_model == 0:
            mbrl.util.common.train_model_and_save_model_and_data(
                dynamics_model,
                model_trainer,
                cfg.overrides,
                replay_buffer,
                work_dir=work_dir,
            )

            # --------- Rollout new model and store imagined trajectories --------
            # Batch all rollouts for the next freq_train_model steps together
            rollout_model_and_populate_sac_buffer(
                model_env,
                replay_buffer,
                agent,
                sac_buffer,
                rollout_length,
                rollout_batch_size,
                cfg.strategy.name == "thompson"
            )

            if debug_mode:
                print(
                    f"Epoch: {epoch}. "
                    f"SAC buffer size: {len(sac_buffer)}. "
                    f"Rollout length: {rollout_length}. "
                    f"Steps: {env_steps}"
                )

        # --------------- Agent Training -----------------
        for _ in range(cfg.overrides.num_sac_updates_per_step):
            use_real_data = rng.random() < cfg.algorithm.real_data_ratio
            which_buffer = replay_buffer if use_real_data else sac_buffer
            if (env_steps + 1) % cfg.overrides.sac_updates_every_steps != 0 or len(
                which_buffer
            ) < cfg.overrides.sac_batch_size:
                break  # only update every once in a while

            agent.update_parameters(
                which_buffer,
                cfg.overrides.sac_batch_size,
                updates_made,
                logger,
                reverse_mask=True,
            )
            updates_made += 1
            if not silent and updates_made % cfg.log_frequency_agent == 0:
                logger.dump(updates_made, save=True)

        env_steps += 1
        obs = next_obs
    return np.float32(0)
