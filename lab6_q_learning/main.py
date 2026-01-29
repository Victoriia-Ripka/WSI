"""
Author: Viktoriia Nowotka
"""
import matplotlib.pyplot as plt
from lab6_q_learning.q_learn import QLearning


def plot_n_steps( nazwa_eksperymentu: str, parametry_uczenia: dict, param_array: list, n_steps_array: list, param_name: str):

    plt.figure()
    plt.plot(param_array, n_steps_array, marker='o')
    plt.xlabel(param_name)
    plt.ylabel("Średnia liczba kroków (n_steps)")

    opis = ", ".join([f"{k}={v}" for k, v in parametry_uczenia.items()])
    plt.title(f"{nazwa_eksperymentu}\n{opis}")

    plt.grid(True)
    plt.show()


def main_test():
    U = [0, 1, 2]
    n_dirs = 4

    e_max = 700
    max_steps = 500

    b = 0.8           # szybkość uczenia β
    y = 0.99          # współczynnik dyskontowania γ
    eps = 0.5         # parametr eksploracji ϵ

    ql = QLearning(U, n_dirs, e_max, max_steps)
    ql.train(b, y, eps)
    n_steps = ql.run()
    print(f"\n\nRun finished after {n_steps} steps")
    print(f"Q-learning alg params: {ql.get_parameters()}")
    ql.close()


def main():
    n_iters_for_averaging_result = 3

    U = [0, 1, 2]
    n_dirs = 4

    e_max = 10
    max_steps = 100

    # eksperyment szybkości uczenia β
    b_array = [0.1, 0.5, 0.9]
    y = 0.99
    eps = 0.5

    n_steps_avg = []

    for b in b_array:
        steps = []

        for _ in range(n_iters_for_averaging_result):
            ql = QLearning(U, n_dirs, e_max, max_steps)
            ql.train(b, y, eps)
            steps.append(ql.run())
            ql.close()

        n_steps_avg.append(sum(steps) / len(steps))

    plot_n_steps(
        nazwa_eksperymentu="Wpływ szybkości uczenia β na zbieżność Q-learning",
        parametry_uczenia={
            "y": y,
            "eps": eps,
            "e_max": e_max,
            "max_steps": max_steps
        },
        param_array=b_array,
        n_steps_array=n_steps_avg,
        param_name="β"
    )

    # eksperyment współczynnik dyskontowania γ
    # współczynnik dyskontowania γ = 1 - 1/H, gdzie H = horyzont kroków.
    # Przy H = 10 => γ = 0.9
    # Przy H = 20 => γ = 0.95
    # Przy H = 40 => γ = 0.975
    # Przy H = 100 => γ = 0.99
    # b = 0.7
    # eps = 0.5
    # y_array = [0.9, 0.95, 0.975, 0.99]
    #
    # n_steps_avg = []
    #
    # for y in y_array:
    #     steps = []
    #
    #     for _ in range(n_iters_for_averaging_result):
    #         ql = QLearning(U, n_dirs, e_max, max_steps)
    #         ql.train(b, y, eps)
    #         steps.append(ql.run())
    #         ql.close()
    #
    #     n_steps_avg.append(sum(steps) / len(steps))
    #
    # plot_n_steps(
    #     nazwa_eksperymentu="Wpływ współczynnika dyskontowania γ na zbieżność Q-learning",
    #     parametry_uczenia={
    #         "b": b,
    #         "eps": eps,
    #         "e_max": e_max,
    #         "max_steps": max_steps
    #     },
    #     param_array=y_array,
    #     n_steps_array=n_steps_avg,
    #     param_name="γ"
    # )

    # eksperyment parametr eksploracji ϵ
    # b = 0.7
    # y = 0.975
    # eps_array = [0.1, 0.5, 0.9]
    #
    # n_steps_avg = []
    #
    # for eps in eps_array:
    #     steps = []
    #
    #     for _ in range(n_iters_for_averaging_result):
    #         ql = QLearning(U, n_dirs, e_max, max_steps)
    #         ql.train(b, y, eps)
    #         steps.append(ql.run())
    #         ql.close()
    #
    #     n_steps_avg.append(sum(steps) / len(steps))
    #
    # plot_n_steps(
    #     nazwa_eksperymentu="Wpływ parametru eksploracji ϵ na zbieżność Q-learning",
    #     parametry_uczenia={
    #         "b": b,
    #         "y": y,
    #         "e_max": e_max,
    #         "max_steps": max_steps
    #     },
    #     param_array=eps_array,
    #     n_steps_array=n_steps_avg,
    #     param_name="ϵ"
    # )


if __name__ == "__main__":
    main()



