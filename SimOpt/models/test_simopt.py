import argparse

import torch
import torch.nn
import gym
from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from sb3_contrib import TRPO
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--episodes', default=50, type=int, help='Number of test episodes')
    parser.add_argument('--render', default=False, action='store_true', help='Render the simulator')
    parser.add_argument('--model', default=None, type=str, help='Model path')
    parser.add_argument('--env', default='target', type=str, help='source/target environment')


    return parser.parse_args()

def main():

    args = parse_args()


    if args.env=='source':
        env = gym.make('CustomHopper-source-v0')
    if args.env=='target':
        env = gym.make('CustomHopper-target-v0')

    gamma = 0.99
    model = TRPO.load(args.model)

    cum_rewards = []
    cum_disc_rewards = []
    for episode in range(args.episodes):
        obs = env.reset()
        test_reward = 0
        done = False
        rewards = []
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            if args.render:
                env_target.render()
            rewards.append(reward)
            test_reward += reward
        print(f"Episode: {episode} | Return: {test_reward}")
        disc_reward = np.sum((np.array(rewards))*(gamma**np.array(range(0, len(rewards)))))
        cum_rewards.append(test_reward)
        cum_disc_rewards.append(disc_reward)

    print(f"Average return: {sum(cum_rewards)/len(cum_rewards)} | Average Disc return: {sum(cum_disc_rewards)/len(cum_disc_rewards)}")
    env.close()


if __name__ == '__main__':
    main()
