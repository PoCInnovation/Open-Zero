import numpy as np
import threading
import multiprocessing
import chess
from typing import Optional
import os
import select

import sys

class chess_game_analyzer:
    def __init__(self):
        self.board = chess.Board()
        self.engine = chess.engine.SimpleEngine.popen_uci("/bins/stockfish")

    def __del__(self):
        await self.engine.quit()

    def run(self):
        readable, _, _ = select.select([ sys.stdin ], [], [], 2.5)
        if len(readable) > 0:
            buffer = sys.stdin.read()

            # Should receive FEN_STRING;EVAL;NEXT_MOVE_IN_UCI
            split_buffer = buffer.split(';')

            self.board = chess.Board(split_buffer[0])
            info = self.engine.analyse(self.board, chess.engine.Limit(depth=20))

            uci_stockfish = info.get("pv")[0].uci
            if uci_stockfish != split_buffer[2]:
                print(split_buffer[0], "; STOCKFISH: [", info.get("score"), "] -> ", \
                info.get("pv")[0].uci, "; OPENZERO: [ ", split_buffer[1], " ] -> ", split_buffer[2])

if __name__ == "__main__":
    analyzer = chess_game_analyzer()

    while True:
        analyzer.run()

