import numpy as np
import threading
import multiprocessing
import chess
import chess.engine
from typing import Optional
import os
import select
import asyncio
import sys
import socket

def get_readable_sockets(sock, client_sockets):
    total_sockets = client_sockets

    client_sockets.append(sock)
    readfds, _, errorfds = select.select(total_sockets, [], total_sockets, 0.5)

    if len(errorfds) > 0:
        for e in errorfds:
            e.close()
            client_sockets.pop(e)

    return readfds

async def main(sock) -> None:
    _, engine = await chess.engine.popen_uci("/bins/stockfish")
    board = chess.Board()
    client_sockets = []

    while not board.is_game_over():
        readables = get_readable_sockets(sock, client_sockets)

        for r in readables:
            if r == sock:
                new_socket, _ = sock.accept()
                client_sockets.append(new_socket)
            else:
                buffer = r.recv(1024)

                if not buffer:
                    break

                # Should receive FEN_STRING;EVAL;NEXT_MOVE_IN_UCI
                split_buffer = buffer.split(';')

                board = chess.Board(split_buffer[0])
                info = engine.analyse(board, chess.engine.Limit(depth=20))

                uci_stockfish = info.get("pv")[0].uci
                if uci_stockfish != split_buffer[2]:
                    print(split_buffer[0], "; STOCKFISH: [", info.get("score"), "] -> ", \
                    info.get("pv")[0].uci, "; OPENZERO: [ ", split_buffer[1], " ] -> ", split_buffer[2])

    sock.close()
    await engine.quit()


if __name__ == "__main__":
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('', 6969))
    sock.listen(5)

    asyncio.set_event_loop_policy(chess.engine.EventLoopPolicy())
    asyncio.run(main(sock))
