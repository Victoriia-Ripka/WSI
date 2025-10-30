# Viktoriia Nowotka
from tkinter import *
import numpy as np
import random
from dataclasses import dataclass

class DotsAndBoxesGame:
    def __init__(self, rows: int, cols: int):
        self.rows = rows          # кількість рядків точок
        self.cols = cols          # кількість колонок точок
        self.horizontal = [[False]*(cols-1) for _ in range(rows)]   # горизонтальні лінії
        self.vertical = [[False]*cols for _ in range(rows-1)]       # вертикальні лінії
        self.current_player = 1
        self.scores = {1: 0, -1: 0}

    def get_valid_moves(self): ...
    def make_move(self, move): ...
    def undo_move(self, move): ...
    def is_game_over(self): ...
    def evaluate(self): ...
    def is_terminal(self): ...
    def make_move(selfself): ...


class Player:
    def __init__(self, char):
        self.char = char


@dataclass
class Move:
    orientation: str   # "H" або "V"
    row: int
    col: int


class MinimaxAgent:
    def __init__(self, depth):
        self.depth = depth

    def choose_move(self, game: TwoPlayerGame): ...

    def minimax(self, node, maximizing_player):
        if depth == 0 or game.is_game_over():
            return node.evaluate()
        if maximizing_player:
            max_eval = float('-inf')
            for move in game.get_valid_moves():
                game.make_move(move)
                eval = self.minimax(game, depth - 1, False)
                game.undo_move(move)
                max_eval = max(max_eval, eval)
            return max_eval
        else:
            min_eval = float('inf')
            for move in game.get_valid_moves():
                game.make_move(move)
                eval = self.minimax(game, depth - 1, True)
                game.undo_move(move)
                min_eval = min(min_eval, eval)
            return min_eval


def main():
    game = DotsAndBoxesGame(3, 3)

    while not game.is_game_over():
        moves = game.get_valid_moves()
        move = random.choice(moves)
        game.make_move(move)
        # if game.current_player == 1:
        #     move = ai.choose_move(game)
        # else:
        #     move = random.choice(game.get_valid_moves())
        # game.make_move(move)
        # print(f"Player {game.current_player} made move {move}")
        winner = game.get_winner()
        if winner is None:
            print(f"Player {game.current_player} made move {move}")
        else:
            print('Winner: Player ' + winner.char)