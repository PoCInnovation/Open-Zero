import numpy as np
import threading
import multiprocessing
import chess
from typing import Optional
import os

import sys

class chess_game_analyzer:
    def __init__(self):
        self.engine = None
        self.board = chess.Board()

        if os.path.isfile("/bins/stockfish"):
           self.engine = chess.engine.SimpleEngine.popen_uci("/bins/stockfish")


    def __del__(self):
        if self.engine != None:
            await self.engine.quit()

    async def analyse_position(self, fen_string, file_handle):
        self.board = Board(fen_string)
        info = self.engine.analyse(self.board, chess.engine.Limit(depth=20))
        print("fen:", fen_string, " - [", info.get("score"),"] -> ", info.get("pv"))
