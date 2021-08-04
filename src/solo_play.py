from chess_a3c import ActorCritic
import chess
import gym
import sys
import socket
import time
import numpy as np

def get_bitboard_from_observation(obs):
    pieces = "pnbrqkPNBRQK"
    result = [['0' for _ in range(8)] for _ in range(8)]

    for i in range(0, 12):
        for board in obs:
            for case in board:
                if board[case] == '1':
                    result[board][case] = pieces[i]
    print (line for line in result)
    return result

def get_fen_from_bitboard(magic_bitboard):
    return magic_bitboard

if __name__ == '__main__':
    network = ActorCritic(7616, 4672)
    env = gym.make('ChessAlphaZero-v0')

    # let's give some time for the server to set up
    time.sleep(1)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(('', 6969))

    observation = env.reset()
    done = False
    reward = 0
    is_white_turn = True
    while not done:
        actions = env.legal_actions
        action = network.choose_action(np.array(observation).flatten(), actions)
        obs, reward, done, _ = env.step(action)
        magic_bitboard = get_bitboard_from_observation(obs)
        fen = get_fen_from_bitboard(magic_bitboard)
        sock.send(bytes(f"{fen};{str(reward)};{env.decode(action)}"))
        is_white_turn = True if not is_white_turn else False

    env.close()
    sock.close()
