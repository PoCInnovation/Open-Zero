import gym
import torch as T
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
import gym_chess
import threading
import multiprocessing
from typing import Optional

import sys

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

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
    def __init__(self, input_dims, n_actions, gamma=0.99):
        super(ActorCritic, self).__init__()

        self.gamma = gamma
        self.output_dims = n_actions

        self.pi1 = nn.Linear(input_dims, 256)
        self.v1 = nn.Linear(input_dims, 256)
        self.pi = nn.Linear(256, n_actions)
        self.v = nn.Linear(256, 1)

        self.rewards = []
        self.actions = []
        self.states = []

    def remember(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def forward(self, state):
        pi1 = F.relu(self.pi1(state))
        v1 = F.relu(self.v1(state))

        pi = self.pi(pi1)
        v = self.v(v1)
        return pi, v

    def calc_R(self, done):
        states = T.tensor(self.states, dtype=T.float)
        #print(states)
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

    def choose_action(self, observation, legal_actions):
        mask = T.zeros(self.output_dims, dtype=bool)
        mask[legal_actions] = True

        state = T.tensor([observation], dtype=T.float)
        pi, v = self.forward(state)
        probs = T.softmax(pi, dim=1)
        dist = CategoricalMasked(probs, mask=mask)
        action = dist.sample().numpy()[0]

        return action

class Agent(mp.Process):
    def __init__(self, w_global_actor_critic, b_global_actor_critic, w_optimizer, b_optimizer, input_dims, n_actions, 
                gamma, lr, name, global_ep_idx, env_id):
        super(Agent, self).__init__()
        self.w_local_actor_critic = ActorCritic(input_dims, n_actions, gamma)
        self.b_local_actor_critic = ActorCritic(input_dims, n_actions, gamma)
        self.w_global_actor_critic = w_global_actor_critic
        self.b_global_actor_critic = b_global_actor_critic
        self.name = 'w%02i' % name
        self.episode_idx = global_ep_idx
        self.env = gym.make(env_id)
        self.w_optimizer = w_optimizer
        self.b_optimizer = b_optimizer

    def run(self):
        t_step = 1
        while self.episode_idx.value < N_GAMES:
            c = 0
            done = False
            observation = self.env.reset()
            w_score = 0
            b_score = 0
            self.w_local_actor_critic.clear_memory()
            self.b_local_actor_critic.clear_memory()
            while not done:
                c += 1
                
                # White turn
                actions = self.env.legal_actions
                action = self.w_local_actor_critic.choose_action(np.array(observation).flatten(), actions)
                #eprint(self.name, action)
                observation_, reward, done, info = self.env.step(action)
                #print(self.env.render(mode='unicode'))
                w_score += reward
                self.w_local_actor_critic.remember(np.array(observation).flatten(), action, reward)
                
                # Black turn
                if not done:
                    actions = self.env.legal_actions
                    action = self.b_local_actor_critic.choose_action(np.array(observation_).flatten(), actions)
                    observation__, reward, done, info = self.env.step(action)
                    #print(self.env.render(mode='unicode'))
                    b_score += -reward
                    self.b_local_actor_critic.remember(np.array(observation_).flatten(), action, -reward)
                    
                #print('curr white score:', w_score, 'curr black score:', b_score, 'agent id:', self.name)

                if t_step % T_MAX == 0 or done:
                    # White backprop
                    loss = self.w_local_actor_critic.calc_loss(done)
                    self.w_optimizer.zero_grad()
                    loss.backward()
                    for local_param, global_param in zip(
                            self.w_local_actor_critic.parameters(),
                            self.w_global_actor_critic.parameters()):
                        global_param._grad = local_param.grad
                    self.w_optimizer.step()
                    self.w_local_actor_critic.load_state_dict(
                            self.w_global_actor_critic.state_dict())
                    self.w_local_actor_critic.clear_memory()

                    # Black backprop
                    if self.b_local_actor_critic.states:
                        #print(self.b_local_actor_critic.states)
                        loss = self.b_local_actor_critic.calc_loss(done)
                        self.b_optimizer.zero_grad()
                        loss.backward()
                        for local_param, global_param in zip(
                            self.b_local_actor_critic.parameters(),
                            self.b_global_actor_critic.parameters()):
                            global_param._grad = local_param.grad
                        self.b_optimizer.step()
                        self.b_local_actor_critic.load_state_dict(
                            self.b_global_actor_critic.state_dict())
                        self.b_local_actor_critic.clear_memory()


                t_step += 1
                observation = observation__
            with self.episode_idx.get_lock():
                self.episode_idx.value += 1
            eprint(self.name, 'episode ', self.episode_idx.value, 'w_reward %.1f' % w_score, 'b_reward %.1f' % b_score, 'steps %d' % c)

if __name__ == '__main__':
    chess = True
    lr = 1e-4
    env_id = 'CartPole-v1' if chess is False else 'ChessAlphaZero-v0'
    n_actions = 2 if chess is False else 4672
    input_dims = [4] if chess is False else 7616
    N_GAMES = 3000
    T_MAX = 5
    w_global_actor_critic = ActorCritic(input_dims, n_actions)
    w_global_actor_critic.share_memory()
    b_global_actor_critic = ActorCritic(input_dims, n_actions)
    b_global_actor_critic.share_memory()
    #optim = SharedAdam(global_actor_critic.parameters(), lr=lr, betas=(0.92, 0.999))
    w_optim = T.optim.Adam(w_global_actor_critic.parameters(), lr=lr)
    b_optim = T.optim.Adam(b_global_actor_critic.parameters(), lr=lr)
    global_ep = mp.Value('i', 0)

    workers = [Agent(w_global_actor_critic,
                    b_global_actor_critic,
                    w_optim,
                    b_optim,
                    input_dims,
                    n_actions,
                    gamma=0.99,
                    lr=lr,
                    name=i,
                    global_ep_idx=global_ep,
                    env_id=env_id) for i in range(mp.cpu_count())]

    [w.start() for w in workers]
    [w.join() for w in workers]
    