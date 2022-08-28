"""Train an RL agent on the OpenAI Gym Hopper environment

TODO: implement 2.2.a and 2.2.b
"""

import torch
import gym
import argparse

from env.custom_hopper import *

from sb3_contrib import TRPO


def main():

	env_s = gym.make('CustomHopper-source-v0')
	env_t = gym.make('CustomHopper-target-v0')

	print(env_s.get_parameters())
	print(env_t.get_parameters())

	model_f = TRPO.load('model_trpo_fixed.zip', print_system_info=False)
	model_r = TRPO.load('model_trpo_random.zip', print_system_info=False)
	

	#### FIXED-SOURCE-SOURCE

	obs_s = env_s.reset()
	tot_reward = 0
	count = 0

	for i in range(10000):
    		action_s, _states_s = model_f.predict(obs_s, deterministic=True)
    		obs_s, reward_s, done_s, info_s = env_s.step(action_s)
    		env_s.render()

    		tot_reward += reward_s 

    		if done_s:
      			obs_s = env_s.reset()
      			count += 1

	env_s.close()
	print('FIXED SOURCE AVG',tot_reward/count)


	#### FIXED-SOURCE-TARGET

	tot_reward = 0
	count = 0
	obs_t = env_t.reset()

	for j in range(10000):
    		action_t, _states_t = model_f.predict(obs_t, deterministic=True)
    		obs_t, reward_t, done_t, info_t = env_t.step(action_t)
    		env_t.render()

    		tot_reward += reward_t 

    		if done_t:
      			obs_t = env_t.reset()
      			count += 1

	env_t.close()
	print('FIXED TARGET AVG',tot_reward/count)

	env_s = gym.make('CustomHopper-source-v0')
	env_t = gym.make('CustomHopper-target-v0')

	#### RANDOM - SOURCE-SOURCE

	obs_s = env_s.reset()
	tot_reward = 0
	count = 0

	for i in range(10000):
    		action_s, _states_s = model_r.predict(obs_s, deterministic=True)
    		obs_s, reward_s, done_s, info_s = env_s.step(action_s)
    		env_s.render()

    		tot_reward += reward_s 

    		if done_s:
      			obs_s = env_s.reset()
      			count += 1

	env_s.close()
	print('SOURCE RANDOM AVG:',tot_reward/count)

	#### RANDOM - TARGET

	tot_reward = 0
	count = 0
	obs_t = env_t.reset()

	for j in range(10000):
    		action_t, _states_t = model_r.predict(obs_t, deterministic=True)
    		obs_t, reward_t, done_t, info_t = env_t.step(action_t)
    		env_t.render()

    		tot_reward += reward_t 

    		if done_t:
      			obs_t = env_t.reset()
      			count += 1

	env_t.close()
	print('TARGET RANDOM AVG',tot_reward/count)

	

if __name__ == '__main__':
	main()