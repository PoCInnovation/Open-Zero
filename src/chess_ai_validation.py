import numpy as np
import threading
import multiprocessing
import chess
from typing import Optional
import os

import sys

class chess_game_analyzer:
    def __init__(self):
        self.board = chess.Board()
        self.engine = chess.engine.SimpleEngine.popen_uci("/bins/stockfish")

    def __del__(self):
        await self.engine.quit()

    async def analyse_position(self, fen_string):
        self.board = chess.Board(fen_string)
        info = self.engine.analyse(self.board, chess.engine.Limit(depth=20))
        print("fen:", fen_string, " - [", info.get("score"),"] -> ", info.get("pv")[0].uci)
