from chess_a3c import ActorCritic
import gym
import sys
import numpy as np
    
if __name__ == '__main__':
    network = ActorCritic(7616, 4672)
    env = gym.make('ChessAlphaZero-v0')

    # @TODO: load model

    observation = env.reset()
    done = False
    reward = 0
    iteration = 0
    while not done:
        actions = env.legal_actions
        action = network.choose_action(np.array(observation).flatten(), actions)
        observation_, reward, done, info = env.step(action)
        iteration += 1

        print(env.render(mode='unicode'), '\n--------\n')

    print(reward, iteration)