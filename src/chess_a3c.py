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
from sharedadam import SharedAdam 
from categoricalmasked import CategoricalMasked 

import asyncio
import chess
import chess.engine
import sys

CUDA = False

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

class ActorCritic(nn.Module):
    def __init__(self, input_dims, n_actions, gamma=0.99):
        super(ActorCritic, self).__init__()

        self.gamma = gamma
        self.output_dims = n_actions

        self.pi1 = nn.Linear(input_dims, 256)
        self.pi2 = nn.Linear(256, 256)
        self.pi3 = nn.Linear(256, 256)
        self.pi4 = nn.Linear(256, 256)
        self.pi5 = nn.Linear(256, 256)
        self.pi6 = nn.Linear(256, 256)
        self.pi7 = nn.Linear(256, 256)
        self.pi8 = nn.Linear(256, 256)
        self.pi9 = nn.Linear(256, 256)
        self.pi10 = nn.Linear(256, 256)
        self.pi = nn.Linear(256, n_actions)
        self.v = nn.Linear(256, 1)

        self.w_rewards = []
        self.w_actions = []
        self.w_states = []

        self.b_rewards = []
        self.b_actions = []
        self.b_states = []

        self.device = T.device('cuda' if T.cuda.is_available() and CUDA == True else 'cpu')

        self.to(self.device)

    def remember(self, color, state, action, reward):
        if color == 'white':
            self.w_states.append(state)
            self.w_actions.append(action)
            self.w_rewards.append(reward)
        elif color == 'black':
            self.b_states.append(state)
            self.b_actions.append(action)
            self.b_rewards.append(reward)
        else:
            print('remember: color unavailable')

    def clear_memory(self):
        self.w_states = []
        self.w_actions = []
        self.w_rewards = []

        self.b_states = []
        self.b_actions = []
        self.b_rewards = []

    def forward(self, state):
        f = F.relu(self.pi1(state))
        f = F.relu(self.pi2(f))
        f = F.relu(self.pi3(f))
        f = F.relu(self.pi4(f))
        f = F.relu(self.pi5(f))
        f = F.relu(self.pi6(f))
        f = F.relu(self.pi7(f))
        f = F.relu(self.pi8(f))
        f = F.relu(self.pi9(f))
        f = F.relu(self.pi10(f))

        pi = self.pi(f)
        v = self.v(f)
        return pi, v

    def calc_R(self, states, rewards, done):
        states = T.tensor(states, dtype=T.float).to(self.device)
        #print(states)
        _, v = self.forward(states)

        R = v[-1]*(1-int(done))

        batch_return = []
        for reward in rewards[::-1]:
            R = reward + self.gamma*R
            batch_return.append(R)
        batch_return.reverse()
        batch_return = T.tensor(batch_return, dtype=T.float).to(self.device)

        return batch_return

    def calc_loss(self, color, done):
        states_ = []
        actions_ = []
        rewards_ = []

        if color == 'white':
            states_ = self.w_states
            actions_ = self.w_actions
            rewards_ = self.w_rewards
        elif color == 'black':
            states_ = self.b_states
            actions_ = self.b_actions
            rewards_ = self.b_rewards

        states = T.tensor(states_, dtype=T.float).to(self.device)
        actions = T.tensor(actions_, dtype=T.float).to(self.device)

        returns = self.calc_R(states_, rewards_, done)

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

        state = T.tensor([observation], dtype=T.float).to(self.device)
        pi, v = self.forward(state)
        probs = T.softmax(pi, dim=1)
        dist = CategoricalMasked(probs, mask=mask)
        action = dist.sample().numpy()[0]

        return action

class Agent(mp.Process):
    def __init__(self, global_actor_critic, optimizer, input_dims, n_actions,
                gamma, lr, name, global_ep_idx, env_id):
        super(Agent, self).__init__()
        self.local_actor_critic = ActorCritic(input_dims, n_actions, gamma)
        self.global_actor_critic = global_actor_critic
        self.name = 'w%02i' % name
        self.episode_idx = global_ep_idx
        self.env = gym.make(env_id)
        self.optimizer = optimizer
        self.render = False

    def run(self):
        N_GAMES = 5000
        T_MAX = 5
        t_step = 1
        while self.episode_idx.value < N_GAMES:
            c = 0
            done = False
            observation = self.env.reset()
            w_score = 0
            b_score = 0
            self.local_actor_critic.clear_memory()
            self.local_actor_critic.clear_memory()
            while not done:
                c += 1

                # White turn
                actions = self.env.legal_actions
                action = self.local_actor_critic.choose_action(np.array(observation).flatten(), actions)
                #print(self.env.decode(action))
                observation_, reward, done, info = self.env.step(action)
                #print(self.env.render(mode='unicode'))
                w_score += reward
                self.local_actor_critic.remember('white', np.array(observation).flatten(), action, reward)

                if done:
                    self.local_actor_critic.b_rewards[len(self.local_actor_critic.b_actions) - 1] = -reward
                    b_score += -reward

                if not done:
                    actions = self.env.legal_actions
                    action = self.local_actor_critic.choose_action(np.array(observation_).flatten(), actions)
                    observation__, reward, done, info = self.env.step(action)
                    #print(self.name, self.env.decode(action))
                    #print(self.env.render(mode='unicode'))
                    b_score += -reward
                    self.local_actor_critic.remember('black', np.array(observation_).flatten(), action, -reward)
                    if done:
                        self.local_actor_critic.w_rewards[len(self.local_actor_critic.w_actions) - 1] = -reward
                        w_score += -reward

                #print('curr white score:', w_score, 'curr black score:', b_score, 'agent id:', self.name)

                if done:
                    # White backprop
                    loss = self.local_actor_critic.calc_loss('white', done)
                    self.optimizer.zero_grad()
                    loss.backward()
                    for local_param, global_param in zip(
                            self.local_actor_critic.parameters(),
                            self.global_actor_critic.parameters()):
                        global_param._grad = local_param.grad
                    self.optimizer.step()
                    self.local_actor_critic.load_state_dict(
                            self.global_actor_critic.state_dict())
                    self.local_actor_critic.clear_memory()

                    # Black backprop
                    if self.local_actor_critic.b_states:
                        #print(self.local_actor_critic.states)
                        loss = self.local_actor_critic.calc_loss('black', done)
                        self.optimizer.zero_grad()
                        loss.backward()
                        for local_param, global_param in zip(
                            self.local_actor_critic.parameters(),
                            self.global_actor_critic.parameters()):
                            global_param._grad = local_param.grad
                        self.optimizer.step()
                        self.local_actor_critic.load_state_dict(
                            self.global_actor_critic.state_dict())
                        self.local_actor_critic.clear_memory()

                t_step += 1
                observation = observation__
            with self.episode_idx.get_lock():
                self.episode_idx.value += 1
                if self.episode_idx.value % 4000 == 0:
                    T.save(self.global_actor_critic.state_dict(), "save.pt")
            eprint(self.name, 'episode ', self.episode_idx.value, 'result %.1f' % reward, 'steps %d' % c)

if __name__ == '__main__':
    mp.set_start_method('spawn')
    lr = 1e-4
    env_id = 'ChessAlphaZero-v0'
    n_actions = 4672
    input_dims =  7616
    global_actor_critic = ActorCritic(input_dims, n_actions)
    try:
        global_actor_critic.load_state_dict(T.load(sys.argv[1]))
    except:
        print("Error while loading model. Program will continue with a fresh model.")
    global_actor_critic.share_memory()
    #optim = SharedAdam(global_actor_critic.parameters(), lr=lr, betas=(0.92, 0.999))
    optim = T.optim.Adam(global_actor_critic.parameters(), lr=lr)
    global_ep = mp.Value('i', 0)



    workers = [Agent(global_actor_critic,
                    optim,
                    input_dims,
                    n_actions,
                    gamma=0.99,
                    lr=lr,
                    name=i,
                    global_ep_idx=global_ep,
                    env_id=env_id) for i in range(mp.cpu_count())]

    workers[0].render = True
    [w.start() for w in workers]
    [w.join() for w in workers]

