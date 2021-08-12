import gui.board
from chess_a3c import ActorCritic
import gym
import chess
import numpy as np

if __name__ == '__main__':
    network = ActorCritic(7616, 4672)
    env = gym.make('ChessAlphaZero-v0')
    board = chess.Board()
    g = gui.board.Board()

    observation = env.reset()
    done = False
    reward = 0
    while not done:
        actions = env.legal_actions
        action = network.choose_action(np.array(observation).flatten(), actions)
        board.push_uci(env.decode(action).uci())
        observation, reward, done, _ = env.step(action)
        g.render_from_board(board)
        if not g.wait_for_input():
            break;
