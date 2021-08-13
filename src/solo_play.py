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

    try:
        network.load_state_dict(T.load(sys.argv[1]))
    except:
        print("Error while loading model. Program will continue with a fresh model.")

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
            if uci_stockfish != env.decode(action):
                print(f'{fen};STOCKFISH:{uci_stockfish};OPENZERO:{env.decode(action)}')
        await asyncio.sleep(1)

    env.close()
    await engine.quit()

if __name__ == '__main__':
    asyncio.set_event_loop_policy(chess.engine.EventLoopPolicy())

    loop = asyncio.get_event_loop()
    loop.run_until_complete(analyse())
