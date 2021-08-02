import numpy as np
import threading
import multiprocessing
import chess
from typing import Optional
import os
import select

import sys
import socket

class chess_game_analyzer:
    def __init__(self):
        self.board = chess.Board()
        self.engine = chess.engine.SimpleEngine.popen_uci("/bins/stockfish")

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind('localhost', 6969)
        self.sock.listen(5)

    def __del__(self):
        self.sock.close()
        await self.engine.quit()

    def run(self):
        readable, _, _ = select.select([ self.sock ], [], [], 2.5)
        if len(readable) > 0:
            buffer = self.sock.recv(1024)

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
        # TODO end loop condition
        analyzer.run()

