from chess_a3c import ActorCritic
import chess
import gym
import sys
import socket
import time
import numpy as np

import torch as T
    
if __name__ == '__main__':
    network = ActorCritic(7616, 4672)
    env = gym.make('ChessAlphaZero-v0')

    # let's give some time for the server to set up
    time.sleep(1)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(('', 6969))

    try:
        network.load_state_dict(T.load(sys.argv[1]))
    except:
        print("Error while loading model. Program will continue with a fresh model.")
    
    observation = env.reset()
    done = False
    reward = 0
    while not done:
        actions = env.legal_actions
        action = network.choose_action(np.array(observation).flatten(), actions)
        board, reward, done, _ = env.step(action)
        sock.send(board.fen(), ";", str(reward), ";", env.decode(action))

