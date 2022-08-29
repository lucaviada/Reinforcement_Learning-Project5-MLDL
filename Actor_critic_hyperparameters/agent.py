import torch
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from utils import discount_rewards

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
        self.fc1_critic = torch.nn.Linear(state_space, self.hidden)
        self.fc2_critic = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_critic_value = torch.nn.Linear(self.hidden, 1)


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
        x_critic = self.tanh(self.fc1_critic(x))
        x_critic = self.tanh(self.fc2_critic(x_critic))
        value = self.fc3_critic_value(x_critic)


        return normal_dist, value


class Agent(object):
    def __init__(self, policy, device='cpu', gamma=0.99, learning_rate=1e-3, opt='adam'):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        if opt=='adam':
            self.optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
        if opt=='sgd':
            self.optimizer = torch.optim.SGD(policy.parameters(), lr=learning_rate)

        self.gamma = gamma
        self.states = []
        self.next_states = []
        self.state_values = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []

    def update_policy(self):
        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device).squeeze(-1)
        states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        next_states = torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
        state_values = torch.stack(self.state_values, dim=0).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        done = torch.Tensor(self.done).to(self.train_device)

        G=[]                         
        for r in range(0, len(rewards)):
            if r != (len(rewards)-1):
                G.append(rewards[r] + self.gamma * state_values[r+1])
            else:
                G.append(rewards[r])
        G = torch.Tensor(G).to(self.train_device) # boostrapped discounted return estimates
        advantage = G - state_values # advantage terms
        advantage_det = advantage.detach()
        actor_loss = torch.sum(-action_log_probs * advantage_det) # actor loss function
        
        G_det = G.detach()
        critic_loss_fn = torch.nn.HuberLoss()
        critic_loss = critic_loss_fn(state_values, G_det) # critic loss function
        
        ac_loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        ac_loss.backward()
        self.optimizer.step()
                                    # gradients computation and step the optimizer

        return

    def get_action(self, state, evaluation=False):
        x = torch.from_numpy(state).float().to(self.train_device)

        normal_dist, state_value = self.policy(x)

        if evaluation:  # Return mean
            return normal_dist.mean, None

        else:   # Sample from the distribution
            action = normal_dist.sample()

            # Compute Log probability of the action [ log(p(a[0] AND a[1] AND a[2])) = log(p(a[0])*p(a[1])*p(a[2])) = log(p(a[0])) + log(p(a[1])) + log(p(a[2])) ]
            action_log_prob = normal_dist.log_prob(action).sum()

            return action, action_log_prob, state_value

    def store_outcome(self, state, next_state, state_value, action_log_prob, reward, done):
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.state_values.append(state_value)
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)

    def reset_outcome(self):
        self.states = []
        self.next_states = []
        self.state_values = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []
