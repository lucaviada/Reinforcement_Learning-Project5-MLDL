
import torch
import gym
import argparse

from env.custom_hopper import *

from sb3_contrib import TRPO


def main():

	env = gym.make('CustomHopper-source-v0')

	print(env.get_parameters())

	model = TRPO("MlpPolicy", env, verbose=1)

	#model = TRPO.load('model_trpo.zip', print_system_info=True)

	model.learn(total_timesteps=1000000)
		

	#print('finito')
	model.save('model_trpo_fixed.zip')
	

	obs = env.reset()

	for i in range(10000):
    		action, _states = model.predict(obs, deterministic=True)
    		obs, reward, done, info = env.step(action)
    		env.render()

    		if done:
      			obs = env.reset()

	env.close()

	

if __name__ == '__main__':
	main()