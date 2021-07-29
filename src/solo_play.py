from chess_a3c import ActorCritic
import gym
import sys
import numpy as np

def get_fen_string(env):
    result = []
    nb_spaces = 0
    pieces = ["RNBQKP"]
    is_black = False

    for i in env.board:
        for j in env.board[i]:
            if env.board[i][j] == 0:
                nb_spaces += 1
                continue
            if env.board[i][j] != 0 and nb_spaces != 0:
                result.append(nb_spaces + 48)
                nb_spaces = 0
                continue

            if env.board[i][j] < 0:
                is_black = True

            piece_value = abs(env.board[i][j]) + 1
            to_append = pieces[piece_value]
            if is_black:
                to_append += 32

            result.append(to_append)
            is_black = False
        result.append('/')
    return result


if __name__ == '__main__':
    network = ActorCritic(7616, 4672)
    env = gym.make('ChessAlphaZero-v0')

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
