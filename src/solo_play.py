from chess_a3c import ActorCritic
import chess
import gym
import sys
import socket
import time
import asyncio
import numpy as np
import torch as T

async def analyse() -> None:
    print('[SERVER] setting up tester...')
    network = ActorCritic(7616, 4672)
    env = gym.make('ChessAlphaZero-v0')
    env_chess = gym.make('Chess-v0')
    board = env_chess.reset()

    try:
        network.load_state_dict(T.load(sys.argv[1]))
    except:
        print("Error while loading model. Program will continue with a fresh model.")
        sys.exit(1)

    observation = env.reset()
    _, engine = await chess.engine.popen_uci("/bins/stockfish")

    print('[SERVER] ready to start...')
    while not env.env.env._board.is_game_over():
        fen = env.env.env._board.fen()
        action = network.choose_action(np.array(observation).flatten(), env.legal_actions)
        env.step(action)

        board = chess.Board(fen)
        with await engine.analyse(board, limit=20) as info:
            uci_stockfish = info.get("pv")[0].uci
            if uci_stockfish != env_chess.decode(action):
                print(f'{fen};STOCKFISH: {info.get("pv")[0].uci};OPENZERO:{env.decode(action)}')

    env_chess.close()
    env.close()
    await engine.quit()

if __name__ == '__main__':
    asyncio.set_event_loop_policy(chess.engine.EventLoopPolicy())
    asyncio.run(analyse())
