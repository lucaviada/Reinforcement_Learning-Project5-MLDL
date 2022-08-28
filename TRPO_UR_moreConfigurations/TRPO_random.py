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

    print(env.get_parameters())

    model = TRPO("MlpPolicy", env, verbose=1)
        
	#model = TRPO.load('model_trpo.zip', print_system_info=True)
    
    bestBounds = []
    
    for conf in range(3):
        for i in range(100):
            
            env.set_random_parameters(conf)
            print(env.get_parameters(), 'for:' , i)
            model.learn(total_timesteps=10000)
            

            #print('finito')
        model.save('model_trpo_random.zip')
        

        obs = env.reset()

        rewards = []
        score = 0
        for i in range(10000):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            score+=reward

            if done:
                obs = env.reset()
                rewards.append(score)
                score = 0

        env.close()
        
        
        bestBounds.append(np.mean(rewards)) #mean of rewards for each configuration
        
    pos, best = np.argmax(bestBounds), max(bestBounds)
    print(pos, best, bestBounds)
    
	

if __name__ == '__main__':
    main()
