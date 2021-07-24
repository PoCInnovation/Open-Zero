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

from gym_chess import ChessEnvV1

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


class ActorCritic(nn.Module):
    def __init__(self, input, actions, gamma=0.99):
        super(ActorCritic, self).__init__()
        self.gamma = gamma
        self.episode_idx = 0
        self.output = actions
        self.env = ChessEnvV1()

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

    def train(self):
        while self.episode_idx < N_GAMES:
            done = False
            observation = self.env.reset()
            score = 0
            self.clear_memory()
            while not done:
                actions = self.env.get_possible_actions()
                action = self.choose_action(observation.flatten(), actions)
                observation_, reward, done, _ = self.env.step(action)
                score += reward
                self.remember(action, observation.flatten(), reward)
                if done:
                    loss = self.calc_loss(done)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                observation = observation_
            self.episode_idx += 1
            print('episode ', self.episode_idx, 'reward %.1f' % score)

if __name__ == '__main__':
    lr = 1e-4
    n_actions = 4098
    input = 64
    N_GAMES = 3000
    model = ActorCritic(input, n_actions)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()