from chess_a3c import ActorCritic
import chess
import gym
import sys
import socket
import time
import numpy as np

if __name__ == '__main__':
    network = ActorCritic(7616, 4672)
    env = gym.make('ChessAlphaZero-v0')

    observation = env.reset()
    done = False
    reward = 0
    while not done:
        fen = env.env.env._board.fen()
        actions = env.legal_actions
        action = network.choose_action(np.array(observation).flatten(), actions)
        _, _, done, _ = env.step(action)
        print(f"{fen};{action}")
        time.sleep(1)

    env.close()
