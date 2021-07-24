from os import name
import torch as T
import torch.nn as nn
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.distributions import Categorical
from torch import einsum
import torch.optim as optim
from torch.optim import optimizer
from torch.optim import Adam


from typing import Optional

import numpy as np

import gym
import gym_chess

from einops import reduce


class CategoricalMasked(Categorical):
    def __init__(self, logits: T.Tensor, mask: Optional[T.Tensor] = None):
        self.mask = mask
        self.batch, self.nb_action = logits.size()
        if mask is None:
            super(CategoricalMasked, self).__init__(logits=logits)
        else:
            self.mask_value = T.finfo(logits.dtype).min
            logits.masked_fill_(~self.mask, self.mask_value)
            super(CategoricalMasked, self).__init__(logits=logits)

    def entropy(self):
        if self.mask is None:
            return super().entropy()
        p_log_p = einsum("ij,ij->ij", self.logits, self.probs)
        p_log_p = T.where(
            self.mask,
            p_log_p,
            T.tensor(0, dtype=p_log_p.dtype, device=p_log_p.device),
        )
        return -reduce(p_log_p, "b a -> b", "sum", b=self.batch, a=self.nb_action)


class SharedAdam(T.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8,
            weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps,
                weight_decay=weight_decay)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = T.zeros_like(p.data)
                state['exp_avg_sq'] = T.zeros_like(p.data)

                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()


class Agents():
    def __init__(self, input, actions):
        super(Agents, self).__init__()
        self.white = ActorCritic(input, actions)
        self.black = ActorCritic(input, actions)
        self.episode_idx = 0
        self.env = gym.make('ChessAlphaZero-v0')
    
    def train(self):
        while self.episode_idx < N_GAMES:
            done = False
            observation = self.env.reset()
            w_score = 0
            b_score = 0
            self.white.clear_memory()
            self.black.clear_memory()
            while not done:
                # White turn
                actions = self.env.legal_actions
                action = self.white.choose_action(observation.flatten(), actions)
                observation_, w_reward, done, _ = self.env.step(action)
                #print('White turn:\n', self.env.render(mode='unicode'))
                w_score += w_reward
                self.white.remember(action, observation.flatten(), w_reward)

                # Black turn
                if not done:
                    actions = self.env.legal_actions
                    action = self.black.choose_action(observation_.flatten(), actions)
                    observation__, b_reward, done, _ = self.env.step(action)
                    #print('Black turn:\n', self.env.render(mode='unicode'))
                    b_score += -b_reward
                    self.black.remember(action, observation_.flatten(), b_reward)


                if done:
                    # White backprop
                    loss = self.white.calc_loss(done)
                    w_optimizer.zero_grad()
                    loss.backward()
                    w_optimizer.step()
                    # Black backprop
                    loss = self.black.calc_loss(done)
                    b_optimizer.zero_grad()
                    loss.backward()
                    b_optimizer.step()

                observation = observation__
            self.episode_idx += 1
            print('episode ', self.episode_idx, 'white reward: %.1f' % w_score, 'black reward: %.1f' % b_score)

class ActorCritic(nn.Module):
    def __init__(self, input, actions, gamma=0.99):
        super(ActorCritic, self).__init__()
        self.gamma = gamma
        self.output = actions

        self.pi1 = nn.Linear(input, 256)
        self.v1 = nn.Linear(input, 256)
        self.pi = nn.Linear(256, actions)
        self.v = nn.Linear(256, 1)

        self.actions = []
        self.rewards = []
        self.states = []

    def forward(self, state):
        pi1 = F.relu(self.pi1(state))
        v1 = F.relu(self.v1(state))

        pi = self.pi(pi1)
        v = self.v(v1)

        return pi, v

    def remember(self, action, state, reward):
        self.actions.append(action)
        self.states.append(state)
        self.rewards.append(reward)

    def clear_memory(self):
        self.actions = []
        self.rewards = []
        self.states = []

    def calc_R(self, done):
        states = T.tensor(self.states, dtype=T.float)
        _, v = self.forward(states)
        R = v[-1]*(1-int(done))

        batch_return = []
        for reward in self.rewards[::-1]:
            R = reward + self.gamma*R
            batch_return.append(R)
        batch_return.reverse()
        batch_return = T.tensor(batch_return, dtype=T.float)

        return batch_return

    def calc_loss(self, done):
        states = T.tensor(self.states, dtype=T.float)
        actions = T.tensor(self.actions, dtype=T.float)

        returns = self.calc_R(done)

        pi, values = self.forward(states)
        values = values.squeeze()
        critic_loss = (returns-values)**2

        probs = T.softmax(pi, dim=1)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)
        actor_loss = -log_probs*(returns-values)

        total_loss = (critic_loss + actor_loss).mean()
    
        return total_loss

    def get_mask(self, possible_moves):
        mask = T.zeros(self.output, dtype=bool)
        mask[possible_moves] = True

        return mask

    def choose_action(self, observation, possible_moves):
        mask = self.get_mask(possible_moves)
        state = T.tensor([observation], dtype=T.float)
        pi, v = self.forward(state)
        probs = T.softmax(pi, dim=1)
        dist = CategoricalMasked(probs, mask=mask)
        action = dist.sample().numpy()[0]
        
        return action

if __name__ == '__main__':
    lr = 1e-4
    n_actions = 4672
    input = 7616
    N_GAMES = 3000
    model = Agents(input, n_actions)
    w_optimizer = optim.Adam(model.white.parameters(), lr=lr)
    b_optimizer = optim.Adam(model.black.parameters(), lr=lr)
    model.train()