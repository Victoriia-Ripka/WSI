from lab2.game.dots_and_boxes import DotsAndBoxes
import random

def main():
    game = DotsAndBoxes()

    while not game.is_finished():
        moves = game.get_moves()
        move = random.choice(moves)
        game.make_move(move)

    winner = game.get_winner()
    if winner is None:
        print('Draw!')
    else:
        print('Winner: Player ' + winner.char)


if __name__ == '__main__':
    main()
