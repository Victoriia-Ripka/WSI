from lab2.game import Player
from lab2.game.dots_and_boxes import DotsAndBoxes
import random

# minimax
# obcinaniem α − β
# Przygotuj program pozwalający użytkownikowi grać z komputerem z możliwością wyboru poziomu trudności.

def main():
    playing_field_rows_size = 2
    playing_field_cols_size = 3
    player1 = Player('1', False)
    player2 = Player('2', False)

    game = DotsAndBoxes(playing_field_rows_size, playing_field_cols_size, player1, player2, 4)
    print(game)

    # запустити 25 разів на полі 5на5, кожен раз з різною глибиною max_ai_depth
    # for i in range(8)
    while not game.is_finished():
        moves = game.get_moves()
        print("Possible moves: ", len(moves))

        if game.get_current_player().ai:
            move = game.choose_best_move(moves)
        else:
            # тут зробити можливість для людини обирати рух
            move = random.choice(moves)
        game.make_move(move)

    winner = game.get_winner()
    if winner is None:
        print('\nDraw!')
    else:
        print('\nWinner: Player ' + winner.char)


if __name__ == '__main__':
    main()
