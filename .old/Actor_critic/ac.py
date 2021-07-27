import gym
import numpy as np
from itertools import count
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.optim import optimizer
from torch.optim.adam import Adam

GAMMA = 0.99
LR = 3e-2
SEED = 543
ENV_NAME = 'CartPole-v0'
LOG_INTERVAL = 10

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

env = gym.make(ENV_NAME)
env.seed(SEED)
torch.manual_seed(SEED)
eps = np.finfo(np.float32).eps.item()

class ActorCritic(nn.Module):
    def __init__(self, input, output):
        super(ActorCritic, self).__init__()
        self.af1 = nn.Linear(input, 256)
        self.action = nn.Linear(256, output)
        self.critic = nn.Linear(256, 1)

        self.saved_actions = []
        self.rewards = []

    def forward(self, obs):
        x1 = F.relu(self.af1(obs))
        action_prob = F.softmax(self.action(x1), dim=-1)
        critic = self.critic(x1)
        return action_prob, critic

    def get_action(self, obs):
        state = torch.from_numpy(obs).float()
        probs, critic = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        self.saved_actions.append(SavedAction(m.log_prob(action), critic))

        return action.item()

    def finish_episode(self):
        R = 0
        saved_actions = self.saved_actions
        policy_losses = []
        value_losses = []
        returns = []

        for r in self.rewards[::-1]:
            R = r + GAMMA * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R - value.item()

            policy_losses.append(-log_prob * advantage)

            value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

        optimizer.zero_grad()

        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

        loss.backward()
        optimizer.step()

        del self.rewards[:]
        del self.saved_actions[:]

    def train(self):
        running_reward = 10

        for i_episode in count(1):
            
            state = env.reset()
            ep_reward = 0

            for t in range(1, 10000):
            
                action = self.get_action(state)
                state, reward, done, _ = env.step(action)
                self.rewards.append(reward)
                ep_reward += reward

                if done:
                    break

            running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
            self.finish_episode()

            if i_episode % LOG_INTERVAL == 0:
                print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(i_episode, ep_reward, running_reward))
            
            if running_reward > env.spec.reward_threshold:
                print("Solved! Running reward is now {} and the last episode runs to {} time steps!".format(running_reward, t))
                break

if __name__ == '__main__':
    input_dims = 4
    output_dims = 2
    model = ActorCritic(input_dims, output_dims)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    model.train()