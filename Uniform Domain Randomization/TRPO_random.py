
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

	for i in range(100):
		env.set_random_parameters()
		print(env.get_parameters(), 'for:' , i)
		model.learn(total_timesteps=10000)
		

	#here changing the domain randomization we have created 3 different configuration

	model.save('models/model_trpo_random_conf1.zip')
	#model.save('models/model_trpo_random_conf2.zip')
	#model.save('models/model_trpo_random_conf3.zip')
	

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