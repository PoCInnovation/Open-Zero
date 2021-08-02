from chess_a3c import ActorCritic
import gym
import sys
import socket
import numpy as np

def build_fen_line(env, i):
    result = []
    nb_spaces = 0
    pieces = ["RNBQKP"]
    is_black = False

    for j in env.board[i]:
        if env.board[i][j] == 0:
            nb_spaces += 1
            continue
        if env.board[i][j] != 0 and nb_spaces != 0:
            result.append(str(nb_spaces))
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


def get_fen_string(env):
    result = []

    for i in env.board:
        result.append(build_fen_line(env, i))
    result[len(result)] = 0
    result.append(' ')
    if env.current_player == env.BLACK:
        result.append('b')
    else:
        result.append('w')

    has_set_a_value = False
    result.append(' ')
    if env.black_king_castle_is_possible:
        has_set_a_value = True
        result.append('K')
    if env.black_queen_castle_is_possible:
        has_set_a_value = True
        result.append('Q')
    if env.white_king_castle_is_possible:
        has_set_a_value = True
        result.append('k')
    if env.white_queen_castle_is_possible:
        has_set_a_value = True
        result.append('q')
    if not has_set_a_value:
        result.append('-')

    # TODO en passant (gym-chess doesn't implement it AFAIK)
    has_set_a_value = False
    result.append(' - ')

    # This should include the number of moves since last take, then total number
    # of moves but gym-chess doesn't support this. it's ok because it's only used
    # to reset clocks in classical (40 mins+ )
    result.append(str(env.move_count))
    result.append(' ')
    result.append(str(env.move_count))

    return result


if __name__ == '__main__':
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(('127.0.0.1', 6969))
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
        sock.send(get_fen_string(env), ";", str(reward), ";", env.decode(action))
        iteration += 1

