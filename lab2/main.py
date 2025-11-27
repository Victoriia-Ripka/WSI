from lab2.game import Player
from lab2.game.dots_and_boxes import DotsAndBoxes
import random
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# TODO Przygotuj program pozwalający użytkownikowi grać z komputerem z możliwością wyboru poziomu trudności.


def main():
    playing_field_rows_size = 5
    playing_field_cols_size = 4
    iters = 25

    ai_depthes_experiments = [
        {'1': 3, '2': 3}, # konrola
        {'1': 3, '2': 1}, # widoczna przewaga
        {'1': 5, '2': 2}, # widoczna przewaga nr2
        {'1': 3, '2': 4}, # nieznaczna rózica
        {'1': 4, '2': 6}, #
        {'1': 1, '2': 5}, # kontrola na dupka
    ]

    for i in range(ai_depthes_experiments.__len__()):
        ai_depth_1 = ai_depthes_experiments[i]['1']
        ai_depth_2 = ai_depthes_experiments[i]['2']

        results_winners = {"1": 0, "2": 0, "Draw": 0}
        history_score_diff = []
        # history_depths = []
        print(f"Starting simulation of {iters} games...")

        for i in range(iters):
            player1 = Player("1", True, ai_depth_1)
            player2 = Player("2", True, ai_depth_2)
            game = DotsAndBoxes(
                playing_field_rows_size, playing_field_cols_size, player1, player2
            )

            while not game.is_finished():
                moves = game.get_moves()
                print("Possible moves: ", len(moves))

                if game.get_current_player().ai:
                    move = game.choose_best_move(moves)
                else:
                    move = random.choice(moves)
                game.make_move(move)

            winner = game.get_winner()
            scores = game.get_scores()

            if winner is None:
                results_winners["Draw"] += 1
                print(f"\nGame {i+1}/{iters}: Draw\n")
            else:
                results_winners[winner.char] += 1
                print(f"\nGame {i+1}/{iters}: Winner {winner.char}\n")

            ai_score = scores.get(player2, 0)
            p1_score = scores.get(player1, 0)
            score_difference = ai_score - p1_score

            history_score_diff.append(score_difference)
            # history_depths.append(ai_depth)

        # make_grafic_random_vs_AI(results_winners, history_depths, history_score_diff)
        make_grafic_AI_vs_AI(
            results_winners,
            history_score_diff,
            iters,
            playing_field_rows_size,
            playing_field_cols_size,
            ai_depth_1,
            ai_depth_2,
        )


def make_grafic_random_vs_AI(winners, depths, score_diffs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Grafika 1: Dystrybucja wygranych
    labels = [
        f'Random (1): {winners["1"]}',
        f'AI (2): {winners["2"]}',
        f'Draw: {winners["Draw"]}',
    ]
    sizes = [winners["1"], winners["2"], winners["Draw"]]
    colors = ["#ff9999", "#66b3ff", "#99ff99"]
    ax1.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
    ax1.set_title("Dystrybucja wygranych")

    # Grafika 2: Ефективність AI залежно від глибини
    ax2.scatter(depths, score_diffs, color="blue", alpha=0.6, label="Wynik gry")
    ax2.xaxis.set_major_locator(MultipleLocator(2))
    ax2.yaxis.set_major_locator(MultipleLocator(2))

    unique_depths = sorted(list(set(depths)))
    avg_scores = []
    for d in unique_depths:
        scores_at_d = [diff for depth, diff in zip(depths, score_diffs) if depth == d]
        avg_scores.append(sum(scores_at_d) / len(scores_at_d))

    ax2.plot(unique_depths, avg_scores, color="red", linewidth=2, label="Średnia marża")
    ax2.axhline(0, color="gray", linestyle="--")
    ax2.set_xlabel("AI głębia")
    ax2.set_ylabel("Różnica wyników (AI - Random)")
    ax2.set_title("Przewaga sztucznej inteligencji kontra głębia")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def make_grafic_AI_vs_AI(
    winners, score_diffs, iters, rows, cols, ai_depth_1, ai_depth_2
):
    game_numbers = list(range(1, iters + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Grafika 1: Dystrybucja wygranych
    labels = [
        f'AI 1 (głębia: {ai_depth_1}): {winners["1"]}',
        f'AI 2 (głębia: {ai_depth_2}): {winners["2"]}',
        f'Remis: {winners["Draw"]}',
    ]
    sizes = [winners["1"], winners["2"], winners["Draw"]]
    colors = ["#ff9999", "#66b3ff", "#99ff99"]
    ax1.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
    ax1.set_title(f"Dystrybucja wygranych. Partii: {iters}. Rozmiar: {rows} x {cols}")

    # Grafika 2: Analiza wyników
    ax2.scatter(
        game_numbers,
        score_diffs,
        color="blue",
        alpha=0.6,
        label="Różnica wyników (AI 1 - AI 2)",
    )
    if score_diffs:
        overall_avg_diff = sum(score_diffs) / len(score_diffs)
    else:
        overall_avg_diff = 0

    ax2.axhline(
        overall_avg_diff,
        color="red",
        linestyle="-",
        linewidth=2,
        label=f"Średnia marża ({overall_avg_diff:.2f})",
    )
    ax2.axhline(0, color="gray", linestyle="--", label="Równowaga (0)")
    ax2.set_xlabel("Numer partii")
    ax2.set_ylabel("Różnica wyników (AI_1 vs AI_2)")
    ax2.set_title(
        f"Przewaga: AI_1 (głębia: {ai_depth_1}) vs AI_2 (głębia: {ai_depth_2})"
    )

    ax2.xaxis.set_major_locator(MultipleLocator(5))
    ax2.yaxis.set_major_locator(MultipleLocator(2))
    ax2.grid(True, which="major", axis="both", linestyle="-", linewidth=0.5)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
