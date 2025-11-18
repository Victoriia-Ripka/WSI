from lab2.game import Player
from lab2.game.dots_and_boxes import DotsAndBoxes
import random
import matplotlib.pyplot as plt

# Przygotuj program pozwalający użytkownikowi grać z komputerem z możliwością wyboru poziomu trudności.

def main():
    playing_field_rows_size = 2
    playing_field_cols_size = 2
    # ai_depth = 5
    iters = 25

    results_winners = {'1': 0, '2': 0, 'Draw': 0}
    history_score_diff = []
    history_depths = []
    print(f"Starting simulation of {iters} games...")

    # запустити 25 разів на полі 5на5, кожен раз з різною глибиною max_ai_depth
    for i in range(iters):
        ai_depth = random.randint(0, playing_field_rows_size - 1)
        player1 = Player('1', False)
        player2 = Player('2', True)
        game = DotsAndBoxes(playing_field_rows_size, playing_field_cols_size, player1, player2, ai_depth)

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
        scores = game.get_scores()

        if winner is None:
            results_winners['Draw'] += 1
            print(f"Game {i+1}/{iters}: Draw")
        else:
            results_winners[winner.char] += 1
            print(f"Game {i+1}/{iters}: Winner {winner.char}")

        ai_score = scores.get(player2, 0)  # або scores[player2] залежно від реалізації
        p1_score = scores.get(player1, 0)
        score_difference = ai_score - p1_score

        history_score_diff.append(score_difference)
        history_depths.append(ai_depth)

    make_grafic(results_winners, history_depths, history_score_diff)


def make_grafic(winners, depths, score_diffs):
        # Створюємо вікно з двома графіками (1 рядок, 2 колонки)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # --- Графік 1: Кругова діаграма перемог ---
        labels = [f'Random (1): {winners["1"]}', f'AI (2): {winners["2"]}', f'Draw: {winners["Draw"]}']
        sizes = [winners['1'], winners['2'], winners['Draw']]
        colors = ['#ff9999', '#66b3ff', '#99ff99']

        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Win Distribution')

        # --- Графік 2: Ефективність AI залежно від глибини ---
        # Сортуємо дані, щоб графік був читабельним (по зростанню глибини)
        sorted_data = sorted(zip(depths, score_diffs))
        sorted_depths, sorted_diffs = zip(*sorted_data)

        # Малюємо scatter plot (точки)
        ax2.scatter(depths, score_diffs, color='blue', alpha=0.6, label='Game Result')

        # Малюємо лінію середнього значення для кожної глибини (трендова лінія)
        unique_depths = sorted(list(set(depths)))
        avg_scores = []
        for d in unique_depths:
            scores_at_d = [diff for depth, diff in zip(depths, score_diffs) if depth == d]
            avg_scores.append(sum(scores_at_d) / len(scores_at_d))

        ax2.plot(unique_depths, avg_scores, color='red', linewidth=2, label='Average Margin')

        # Лінія нуля (нічия по очках)
        ax2.axhline(0, color='gray', linestyle='--')

        ax2.set_xlabel('AI Depth')
        ax2.set_ylabel('Score Difference (AI - Random)')
        ax2.set_title('AI Advantage vs Depth')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()
