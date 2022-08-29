#this file is based on an adaptation from the source written in https://github.com/DLR-RM/rl-baselines3-zoo

import optuna
import torch.nn
import torch
import torch.nn as nn
import gym
import argparse

from env.custom_hopper import *
from sb3_contrib import TRPO
from utilss import linear_schedule

def objective(trial):
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512])
    n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1)
    lr_schedule = "constant"
    n_critic_updates = trial.suggest_categorical("n_critic_updates", [5, 10, 20, 25, 30])
    cg_max_steps = trial.suggest_categorical("cg_max_steps", [5, 10, 20, 25, 30])
    target_kl = trial.suggest_categorical("target_kl", [0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0.001])
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
    net_arch = trial.suggest_categorical("net_arch", ["small", "medium"])
    ortho_init = False
    activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])

    if batch_size > n_steps:
        batch_size = n_steps

    if lr_schedule == "linear":
        learning_rate = linear_schedule(learning_rate)

    net_arch = {
        "small": [dict(pi=[64, 64], vf=[64, 64])],
        "medium": [dict(pi=[256, 256], vf=[256, 256])],
    }[net_arch]

    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[activation_fn]

    try:
        env = gym.make('CustomHopper-source-v0')
        env.reset()
        model = TRPO("MlpPolicy", env, batch_size = batch_size, n_steps = n_steps, gamma = gamma, learning_rate = learning_rate, n_critic_updates = n_critic_updates, cg_max_steps = cg_max_steps, target_kl = target_kl, gae_lambda = gae_lambda, policy_kwargs = dict(
                    log_std_init=-2,
                    ortho_init=False,
                    activation_fn=activation_fn,
                    net_arch=net_arch
                    ), verbose = 0)
        model.learn(total_timesteps=100000, log_interval=4)
        obs = env.reset()
        train_reward = 0
        counter = 0
        rew = []
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            #env.render()
            train_reward += reward
            if done:
                obs = env.reset()
                counter += 1
                rew.append(train_reward)
                train_reward = 0
                if counter == 100:
                    break

        return sum(rew)/len(rew)
    except:
        return -1

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=None, timeout=None)
study.best_params
print(study.best_trial.value)  # Show the best value.
