"""Test an RL agent on the OpenAI Gym Hopper environment"""

import torch
import gym
import argparse
import numpy as np

from env.custom_hopper import *
from agent import Agent, Policy

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None, type=str, help='Model path')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--render', default=False, action='store_true', help='Render the simulator')
    parser.add_argument('--episodes', default=50, type=int, help='Number of test episodes')
    parser.add_argument('--env', default='target', type=str, help='Test environment [source, target]')

    return parser.parse_args()

args = parse_args()


def main():

    if args.env=='source':
    	env = gym.make('CustomHopper-source-v0')
    if args.env=='target':
    	env = gym.make('CustomHopper-target-v0')

    gamma = 0.99

    print('Action space:', env.action_space)
    print('State space:', env.observation_space)
    print('Dynamics parameters:', env.get_parameters())

    observation_space_dim = env.observation_space.shape[-1]
    action_space_dim = env.action_space.shape[-1]

    policy = Policy(observation_space_dim, action_space_dim)
    policy.load_state_dict(torch.load(args.model), strict=True)

    agent = Agent(policy, device=args.device)

    cum_rewards = []
    cum_disc_rewards = []
    for episode in range(args.episodes):
        done = False
        test_reward = 0
        state = env.reset()
        rewards = []
        while not done:

            action, _ = agent.get_action(state, evaluation=True)

            state, reward, done, info = env.step(action.detach().cpu().numpy())

            rewards.append(reward)
            if args.render:
                env.render()

            test_reward += reward

        disc_reward = np.sum((np.array(rewards))*(gamma**np.array(range(0, len(rewards)))))
        print(f"Episode: {episode} | Return: {test_reward} | Disc return: {disc_reward}")
        cum_rewards.append(test_reward)
        cum_disc_rewards.append(disc_reward)

    print(f"Average return: {sum(cum_rewards)/len(cum_rewards)} | Average Disc return: {sum(cum_disc_rewards)/len(cum_disc_rewards)}")

if __name__ == '__main__':
	main()
