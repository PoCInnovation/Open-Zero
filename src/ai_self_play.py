import gui.board
from chess_a3c import ActorCritic
import gym
import chess
import numpy as np



if __name__ == '__main__':
    g = gui.board.Board()
    g.menu(g)

def start_game(g):
    network = ActorCritic(7616, 4672)
    env = gym.make('ChessAlphaZero-v0')
    board = chess.Board()
    turn = 0
    observation = env.reset()
    done = False
    while not done:
        if (g.player_color == "Black" or turn > 0):
            actions = env.legal_actions
            action = network.choose_action(np.array(observation).flatten(), actions)
            board.push_uci(env.decode(action).uci())
            observation, _, done, _ = env.step(action)
        g.render_from_board(board)
        player_action = g.wait_for_input(board, env.legal_actions, env)
        observation, _, done, _ = env.step(player_action)
        g.render_from_board(board)
        turn += 1
        if not player_action:
            break
