#con batch
import torch
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from classes_actcrit_batch.utils import discount_rewards

class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.tanh = torch.nn.Tanh()

        """
            Actor network
        """
        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)
        
        # Learned standard deviation for exploration at training time 
        self.sigma_activation = F.softplus
        init_sigma = 0.5
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space)+init_sigma)
    

        """
            Critic network
        """
        # TODO 2.2.b: critic network for actor-critic algorithm
        #estimated value function
        self.fc1_critic = torch.nn.Linear(state_space, self.hidden)
        self.fc2_critic= torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_output = torch.nn.Linear(self.hidden, 1)
    
    
    
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        """
            Actor
        """
        x_actor = self.tanh(self.fc1_actor(x))
        x_actor = self.tanh(self.fc2_actor(x_actor))
        action_mean = self.fc3_actor_mean(x_actor)

        sigma = self.sigma_activation(self.sigma)
        normal_dist = Normal(action_mean, sigma)


        """
            Critic
        """
        # TODO 2.2.b: forward in the critic network

        x_critic = self.tanh(self.fc1_critic(x))
        x_critic = self.tanh(self.fc2_critic(x_critic))
        output = self.fc3_output(x_critic)
        
        return normal_dist, output


class Agent(object):
    def __init__(self, policy, device='cpu', nBatches = 10):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
       
        self.nBatches = nBatches
        self.gamma = 0.99 
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.output = []
        self.done = []

        

    def update_policy(self):
        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device).squeeze(-1)
        states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        next_states = torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        values = torch.stack(self.output, dim = 0).to(self.train_device).squeeze(-1)
        done = torch.Tensor(self.done).to(self.train_device) 
       
        
       
        lossBatches = []
 
        for j in range(0, len(values), self.nBatches): #indica l'indice in cui inizia il batch
            g = []
            
            valuesBatch = values[j:j+ self.nBatches]
            rewardsBatch = rewards[j:j + self.nBatches]
            
            
            for el in range(0, len(valuesBatch)):
            
                if (el!=(len(valuesBatch)-1)):
                    g.append(rewards[el]+self.gamma*values[el+1]) #stimo il return
                else:
                    g.append(rewards[el])
            g = torch.Tensor(g).to(self.train_device)
            g = g.detach()
            adv = g - values[j:(j+ self.nBatches)] 
    
            
            adv_det = adv.detach()
            actor_loss = sum(-adv_det * action_log_probs[j:(j + self.nBatches)])  
                
            loss_fn = torch.nn.HuberLoss()
            critic_loss = loss_fn(values[j:(j + self.nBatches)], g) 
               
                
            loss = (actor_loss+critic_loss)
            lossBatches.append(loss)
                
            
            
           
        
                #--------------------------------------------------------compute gradients and step the optimizer
        m = 0
        count = 0
        
        for el in lossBatches:
            count+=1
            m+=el
        meanLoss = m/count
        
        self.optimizer.zero_grad()
        meanLoss.backward()
        self.optimizer.step()
              
     

        return        

    def get_action(self, state, evaluation=False): 
        x = torch.from_numpy(state).float().to(self.train_device)

        #policy= NN method
        normal_dist, output = self.policy(x)

        if evaluation:  # Return mean
            return normal_dist.mean, None 

        else:   # Sample from the distribution
            action = normal_dist.sample()

            # Compute Log probability of the action [ log(p(a[0] AND a[1] AND a[2])) = log(p(a[0])*p(a[1])*p(a[2])) = log(p(a[0])) + log(p(a[1])) + log(p(a[2])) ]
            action_log_prob = normal_dist.log_prob(action).sum()

            return action, action_log_prob, output

    def store_outcome(self, state, next_state, action_log_prob, reward, output, done):
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))  
        self.output.append(torch.Tensor(output)) 
        self.done.append(done)
        
    def reset(self):
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.output = []
        self.done = []
    

