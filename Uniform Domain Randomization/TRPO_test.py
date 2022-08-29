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

	print('Configuration 1')

	model_f = TRPO.load('models/model_trpo_fixed.zip', print_system_info=False)
	model_r = TRPO.load('models/model_trpo_random_conf1.zip', print_system_info=False)
	
	## with FIXED is without randomization of masses, with RANDOM is the UDR


	#### FIXED-SOURCE-SOURCE

	obs_s = env_s.reset()
	tot_reward = 0
	count = 0
	gamma=0.99 #
	tot_disc_reward = [] 
	rewards = [] 

	for i in range(20000):
    		action_s, _states_s = model_f.predict(obs_s, deterministic=True)
    		obs_s, reward_s, done_s, info_s = env_s.step(action_s)
    		#env_s.render()

    		tot_reward += reward_s
    		rewards.append(reward_s) 

    		if done_s:
      			obs_s = env_s.reset()
      			count += 1
      			disc_rewards = np.sum((np.array(rewards))*(gamma**np.array(range(0,len(rewards)))))
      			tot_disc_reward.append(disc_rewards) 
      			rewards = [] 

	env_s.close()
	print('SOURCE FIXED AVG',tot_reward/count)
	print('DISCOUNTED AVG', sum(tot_disc_reward)/count) 
	print('EPISODES:', count)


	#### FIXED-SOURCE-TARGET

	tot_reward = 0
	count = 0
	tot_disc_reward = [] 
	rewards = [] 
	obs_t = env_t.reset()

	for j in range(20000):
    		action_t, _states_t = model_f.predict(obs_t, deterministic=True)
    		obs_t, reward_t, done_t, info_t = env_t.step(action_t)
    		#env_t.render()

    		tot_reward += reward_t
    		rewards.append(reward_s) 

    		if done_t:
      			obs_t = env_t.reset()
      			count += 1
      			disc_rewards = np.sum((np.array(rewards))*(gamma**np.array(range(0,len(rewards)))))
      			tot_disc_reward.append(disc_rewards) 
      			rewards = [] 

	env_t.close()
	print('TARGET FIXED AVG',tot_reward/count)
	print('DISCOUNTED AVG', sum(tot_disc_reward)/count)
	print('EPISODES:', count)


	env_s = gym.make('CustomHopper-source-v0')
	env_t = gym.make('CustomHopper-target-v0')

	#### RANDOM - SOURCE-SOURCE

	obs_s = env_s.reset()
	tot_reward = 0
	count = 0
	tot_disc_reward = [] 
	rewards = [] 

	for i in range(20000):
    		action_s, _states_s = model_r.predict(obs_s, deterministic=True)
    		obs_s, reward_s, done_s, info_s = env_s.step(action_s)
    		#env_s.render()

    		tot_reward += reward_s
    		rewards.append(reward_s) 

    		if done_s:
      			obs_s = env_s.reset()
      			count += 1
      			disc_rewards = np.sum((np.array(rewards))*(gamma**np.array(range(0,len(rewards)))))
      			tot_disc_reward.append(disc_rewards) 
      			rewards = [] 

	env_s.close()
	print('SOURCE RANDOM AVG:',tot_reward/count)
	print('DISCOUNTED AVG', sum(tot_disc_reward)/count)
	print('EPISODES:', count)


	#### RANDOM - TARGET

	tot_reward = 0
	count = 0
	tot_disc_reward = [] 
	rewards = [] 
	obs_t = env_t.reset()

	for j in range(20000):
    		action_t, _states_t = model_r.predict(obs_t, deterministic=True)
    		obs_t, reward_t, done_t, info_t = env_t.step(action_t)
    		#env_t.render()

    		tot_reward += reward_t
    		rewards.append(reward_s) 

    		if done_t:
      			obs_t = env_t.reset()
      			count += 1
      			disc_rewards = np.sum((np.array(rewards))*(gamma**np.array(range(0,len(rewards)))))
      			tot_disc_reward.append(disc_rewards) 
      			rewards = [] 

	env_t.close()
	print('TARGET RANDOM AVG',tot_reward/count)
	print('DISCOUNTED AVG', sum(tot_disc_reward)/count)
	print('EPISODES:', count)


	

if __name__ == '__main__':
	main()