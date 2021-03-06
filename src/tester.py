from chess_a3c import ActorCritic
import chess
import chess.engine
import gym
import sys
import numpy as np
import torch as T

def analyse(fen_str, engine):
    board = chess.Board(fen=fen_str)
    info = engine.analyse(board, chess.engine.Limit(time=.5))
    return info["pv"][0]

if __name__ == '__main__':
    network = ActorCritic(7616, 4672)
    env = gym.make('ChessAlphaZero-v0')
    engine = chess.engine.SimpleEngine.popen_uci("/bins/stockfish")
    try:
        network.load_state_dict(T.load(sys.argv[1]))
    except:
        print("[WARNING] Error while loading model. Program will continue with a fresh model.")
    observation = env.reset()
    while not env.env.env._board.is_game_over():
        fen = env.env.env._board.fen()
        action = network.choose_action(np.array(observation).flatten(), env.legal_actions)
        env.step(action)
        stockfish_move = analyse(fen, engine)
        try:
            open_zero_uci = env.decode(action)
            if stockfish_move.uci() != open_zero_uci:
                print(f'{fen};STOCKFISH:{stockfish_move.uci()};OPENZERO:{open_zero_uci}')
        except:
            print('[WARNING] Could not decode action.')
    env.close()
    engine.quit()
