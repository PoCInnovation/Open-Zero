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

    async def analyse_position(self):

        readable, _, _ = select.select([ sys.stdin ], [], [], 2.5)

        if len(readable) > 0:
            buffer = sys.stdin.read()

            # Should receive FEN_STRING;EVAL;NEXT_MOVE_IN_UCI
            split_buffer = buffer.split(';')

            self.board = chess.Board(split_buffer[0])
            info = self.engine.analyse(self.board, chess.engine.Limit(depth=20))
            print("fen:", split_buffer[0], " -- STOCKFISH: [", info.get("score"),"] -> ", \
            info.get("pv")[0].uci, " -- OPENZERO: [ ", split_buffer[1], " ] -> ", split_buffer[2])
