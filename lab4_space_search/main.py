# Viktoriia Nowotka
from gradient_descent import GradientDescent
from charts import *


def main():
    r_min, r_max = -32.768, 32.768
    max_iter = 200
    # TODO zbadać wpływ rozmiaru kroku na działanie algorytmu
    steps = [0.01, 0.02, 0.05, 0.075, 0.1] # 0.005, 0.01, 0.015, 0.02, 0.05, 0.075, 0.09, 0.1, 0.15, 0.2, 0.25
    # TODO zbadać wpływ punktu początkowego
    x0 = 23.0
    y0 = 0.0

    history_1d = {}
    final_values_1d = []

    # 1D Wykres funkcji
    visualization_ackley_function(r_min, r_max)

    print("Zbieranie danych dla ackley_function...")
    for step in steps:
        gradient_descent = GradientDescent(r_min, r_max, step, max_iter)
        x, history = gradient_descent.solve(ackley_function, x0, y0)
        history_1d[step] = history
        result = ackley_function(x[0], x[1])
        final_values_1d.append(result)
        print(gradient_descent.get_parameters())
        print(f"final value: {np.round(result, 2)}, punkt: {np.round(x[0], 2)}")

    # 1D Wykres Zbieżności
    visualization_convergence_ackley_function(history_1d)

    # 1D Wykres trajektoria minimalizacji z najlepszym krokiem
    best_index = np.argmin(final_values_1d)
    optimal_step = steps[best_index]
    gradient_descent = GradientDescent(r_min, r_max, optimal_step, max_iter)
    final_pos, history_f, path_x, path_y = gradient_descent.solve_with_trajectory(ackley_function, x0, y0)
    visualize_best_step_2d(final_pos, history_f, path_x, optimal_step)

    # TODO zbadać wpływ rozmiaru kroku na działanie algorytmu
    steps = [0.15, 0.2, 0.25, 0.3, 0.35]
    # TODO zbadać wpływ punktu początkowego
    x0 = 10.0
    y0 = 10.0
    history_2d = {}
    final_values_2d = []

    # 2D Wykres funkcji
    visualization_ackley_function_3d(r_min, r_max)

    print("\nZbieranie danych dla two_dimensional_ackley_function...")
    for step in steps:
        gradient_descent = GradientDescent(r_min, r_max, step, max_iter)
        x, history = gradient_descent.solve(ackley_function_2d, x0, y0)
        history_2d[step] = history
        result = ackley_function_2d(x[0], x[1])
        final_values_2d.append(result)
        print(gradient_descent.get_parameters())
        print(f"final value: {np.round(result, 2)}, punkt: ({np.round(x[0], 2)};{np.round(x[0], 2)})")

    # 2D Wykres Zbieżności
    visualization_convergence_ackley_function_2d(history_2d)

    # 2D Wykres trajektoria minimalizacji z najlepszym krokiem
    best_index = np.argmin(final_values_2d)
    optimal_step = steps[best_index]
    gradient_descent = GradientDescent(r_min, r_max, optimal_step, max_iter)
    final_pos, history_f, path_x, path_y = gradient_descent.solve_with_trajectory(ackley_function_2d, x0, y0)
    visualize_best_step_3d(final_pos, history_f, path_x, path_y, optimal_step)


if __name__ == "__main__":
    main()
