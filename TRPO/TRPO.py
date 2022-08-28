"""Train an RL agent on the OpenAI Gym Hopper environment

TODO: implement 2.2.a and 2.2.b
"""

import torch
import gym
import argparse

from env.custom_hopper import *

from sb3_contrib import TRPO


def main():

	env = gym.make('CustomHopper-source-v0')

	model = TRPO("MlpPolicy", env, verbose=1)
	model.learn(total_timesteps=100000)

	obs = env.reset()
	for i in range(1000):
    		action, _states = model.predict(obs, deterministic=True)
    		obs, reward, done, info = env.step(action)
    		env.render()

    		if done:
      			obs = env.reset()

	env.close()

	

if __name__ == '__main__':
	main()