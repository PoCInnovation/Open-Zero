import gym
import gym_chess
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
from gym_chess import ChessEnvV1
from typing import Optional

GAMMA = 0.99
LR = 3e-2
LOG_INTERVAL = 10

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
torch.autograd.set_detect_anomaly(True)
env = ChessEnvV1()
eps = np.finfo(np.float32).eps.item()

class CategoricalMasked(Categorical):

    def __init__(self, logits: torch.Tensor, mask: Optional[torch.Tensor] = None):
        self.mask = mask
        self.batch, self.nb_action = logits.size()
        if mask is None:
            super(CategoricalMasked, self).__init__(logits=logits)
        else:
            self.mask_value = torch.finfo(logits.dtype).min
            logits.masked_fill_(~self.mask, self.mask_value)
            super(CategoricalMasked, self).__init__(logits=logits)

class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorCritic, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.pi1 = nn.Linear(input_dim, 128)
        self.v1 = nn.Linear(input_dim, 128)
        self.pi2 = nn.Linear(128, output_dim)
        self.v2 = nn.Linear(128, 1)

        self.saved_actions = []
        self.rewards = []

    def forward(self, state):
        pi1 = F.relu(self.pi1(state))
        v1 = F.relu(self.v1(state))

        pi = self.pi2(pi1)
        v = self.v2(v1)

        return pi, v

#fonction pour récupérer les masques en fonction des moves
    def get_mask(self, possible_moves):
        mask = torch.zeros(self.output_dim, dtype=bool)
        mask[possible_moves] = True

        return mask

#choose_action modifié pour fonctionner avec les masques
    def choose_action(self, observation, possible_moves):
        mask = self.get_mask(possible_moves)
        state = torch.tensor([observation], dtype=torch.float)
        pi, v = self.forward(state)
        probs = torch.softmax(pi, dim=1)
        dist = CategoricalMasked(probs, mask=mask)
        action = dist.sample().numpy()[0]
        self.saved_actions.append(SavedAction(dist.log_prob(torch.tensor(action)), v))
        
        return action

    # def get_action(self, obs):
    #     state = torch.from_numpy(obs).float()
    #     probs, critic = self.forward(state)
    #     m = Categorical(probs)
    #     action = m.sample()
    #     self.saved_actions.append(SavedAction(m.log_prob(action), critic))

        return action.item()

    def finish_episode(self):
        R = 0
        saved_actions = self.saved_actions
        policy_losses = []
        value_losses = []
        returnss = []

        for r in self.rewards[::-1]:
            R = r + GAMMA * R
            returnss.insert(0, R)

        returns = torch.tensor(returnss)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R - value.item()

            policy_losses.append(-log_prob * advantage)
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([[R]])))


        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        del self.rewards[:]
        del self.saved_actions[:]

    def train(self):
        running_reward = 10

        for i_episode in count(1):
            
            state = env.reset()
            ep_reward = 0
            done = False

            while not done:
                #pour récup les actions                
                actions = env.get_possible_actions()
                action = self.choose_action(state.flatten(), actions)
                state, reward, done, _ = env.step(action)
                self.rewards.append(reward)
                ep_reward += reward

            print("episode fini")
            running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
            self.finish_episode()

            if i_episode % LOG_INTERVAL == 0:
                print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(i_episode, ep_reward, running_reward))
            
            if running_reward > env.spec.reward_threshold:
                print("Solved! Running reward is now {} and the last episode runs to {} time steps!".format(running_reward, t))
                break

if __name__ == '__main__':
    input_dims = 64
    output_dims = 4098
    model = ActorCritic(input_dims, output_dims)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    model.train()