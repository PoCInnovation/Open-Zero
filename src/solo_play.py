from chess_a3c import ActorCritic
import chess
import chess.engine
import gym
import sys
import numpy as np
import torch as T

def analyse(fen):
    engine = chess.engine.SimpleEngine.popen_uci("/bins/stockfish")
    info = engine.analyse(chess.Board(fen), chess.engine.Limit(time=1.5, depth=20))
    engine.quit()
    return info["pv"][0]

if __name__ == '__main__':
    print('[LOG] Setting up env...')
    network = ActorCritic(7616, 4672)
    env = gym.make('ChessAlphaZero-v0')

    print('[LOG] Set up gym-env.')
    try:
        network.load_state_dict(T.load(sys.argv[1]))
    except:
        print("[WARNING] Error while loading model. Program will continue with a fresh model.")

    observation = env.reset()
    while not env.env.env._board.is_game_over():
        print('[LOG] board not done.')
        fen = env.env.env._board.fen()
        action = network.choose_action(np.array(observation).flatten(), env.legal_actions)
        env.step(action)
        print(f'[LOG] Just played {action}')

        stockfish_move = analyse(fen)
        print(f'[LOG] Analysis:{stockfish_move.uci()}')
        try:
            open_zero_uci = env.decode(action)
            if stockfish_move.uci() != open_zero_uci:
                print(f'{fen};STOCKFISH:{stockfish_move.uci()};OPENZERO:{open_zero_uci}')
        except:
            print('[WARNING] Could not parse move {action}.')
    env.close()
