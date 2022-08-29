import torch.nn
import torch
import torch.nn as nn
import gym
import argparse

from env.custom_hopper import *
from sb3_contrib import TRPO
from stable_baselines3.ppo.policies import MlpPolicy

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--env', default='source', type=str, help='Train environment [source, target]')

    return parser.parse_args()

args = parse_args()

if args.env=='source':
    env = gym.make('CustomHopper-source-v0')
if args.env=='target':
    env = gym.make('CustomHopper-target-v0')


model = PPO("MlpPolicy", env, verbose = 1, device = args.device)

model.learn(total_timesteps=1000000)
if args.env=='source':
    model_name = 'modelPPO_source.zip'
if args.env=='target':
    model_name = 'modelPPO_target.zip'
    
model.save(model_name)
env.close()
