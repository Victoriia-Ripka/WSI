# Viktoriia Nowotka
import numpy as np
from calc import calc_target
from solver import Solver
from typing import Any, Callable, Dict
import matplotlib.pyplot as plt

plt.style.use('_mpl-gallery')

def find_best(population: np.ndarray, O: np.ndarray):
    best_ind = np.argmax(O)
    return population[best_ind], O[best_ind]


def evaluation(func: Callable[[np.ndarray], float], population: np.ndarray):
    return np.array([func(ind) for ind in population])


class GenAlg(Solver):
    def __init__(self, m: int = 300, t_max: int = 300, genome_length: int = 200, p_c: float = 0.9, p_m: float = 0.1):
        self.m = m
        self.t_max = t_max
        self.genome_length = genome_length
        self.p_c = p_c
        self.p_m = p_m


    def get_parameters(self) -> Dict[str, Any]:
        return {'m': self.m, 't_max': self.t_max, 'p_c': self.p_c, 'p_m': self.p_m}


    def create_population(self):
        return np.random.randint(0, 2, (self.m, self.genome_length*2))

    # selekcja ruletkowa
    def selection(self, population: np.ndarray, O: np.ndarray):
        if np.max(O) == np.min(O):
            scaled = np.ones_like(O)
        elif np.min(O) < 0:
            shifted_O = O - np.min(O)
            O_min = np.min(shifted_O)
            O_max = np.max(shifted_O)
            scaled = (shifted_O - O_min) / (O_max - O_min)
        else:
            scaled = O

        probs = scaled / (np.sum(scaled))
        selected_indices = np.random.choice(len(population), size=self.m, p=probs)
        selected = np.array([population[selected_indic] for selected_indic in selected_indices])
        return selected


    # krzyżowanie jednopunktowe
    def crossover(self, p1: np.ndarray, p2: np.ndarray):
        c1, c2 = np.copy(p1), np.copy(p2)

        if np.random.random() <= self.p_c:
            p = np.random.randint(1, len(p1) - 2)
            c1[p:], c2[p:] = p2[p:], p1[p:]

        return c1, c2


    def crossover_population(self, population: np.ndarray):
        new_pop = []
        # np.random.shuffle(population)
        for i in range(0, len(population), 2):
            if i + 1 < len(population):
                p1, p2 = population[i], population[i + 1]
                c1, c2 = self.crossover(p1, p2)
                new_pop.append(c1)
                new_pop.append(c2)
            else:
                new_pop.append(population[i])
        return np.array(new_pop)


    # mutacja
    def mutate(self, population: np.ndarray):
        pop = np.copy(population)
        for i in range(len(pop)):
            for j in range(pop[i].shape[0]):
                if np.random.random() <= self.p_m:
                    pop[i][j] = 1 - pop[i][j]
        return pop


    def solve(self, problem: Callable[[np.ndarray], float], x0, *args, **kwargs):
        t = 0
        population = self.create_population()
        O = evaluation(problem, population)
        x_best, o_best = find_best(population, O)
        history = [o_best]

        while t < self.t_max and o_best < -0.01:
            selected_population = self.selection(population, O)
            crossed_population = self.crossover_population(selected_population)
            mutated_population = self.mutate(crossed_population)
            O_t = evaluation(problem, mutated_population)
            x_t, o_t = find_best(mutated_population, O_t)
            history.append(o_best)

            if o_t > o_best:
                x_best, o_best = x_t, o_t

            if t % 50 == 0 or t == self.t_max - 1:
                print(f"Najlepsze w iteracji {t}:", np.round(o_best, 3))

            # sukcesja generacyjna
            population = mutated_population
            O = O_t
            t += 1

        return x_best, o_best, history, t


if __name__ == '__main__':
    starts_number = 25
    t_max = 200
    q_x = calc_target
    genome_length = 200

    # hiperparametry
    m = 250 # 50, 100, 150, 200, 250*,  300
    pc = 0.7
    pm = 0.002 # 0.002*, 0.02, 0.1

    solver = GenAlg(m, t_max, genome_length, pc, pm)
    values = []
    solutions = []
    fitness_history = []
    iterations_number = []

    print(f"Uruchomienie optymalizacji...")
    for i in range(starts_number):
        print(f"\nUruchomienie algorytmu: {i+1}")
        x, o, history, t = solver.solve(problem=q_x, x0=None)
        values.append(x)
        solutions.append(o)
        fitness_history.append(history)
        iterations_number.append(t)

    solutions = np.array(solutions)
    values = np.array(values)
    iterations_number = np.array(iterations_number)

    mean = np.mean(solutions)
    std = np.std(solutions)
    best_idx = np.argmax(solutions)
    best_fitness = solutions[best_idx]
    best_solution = values[best_idx]
    worst_idx = np.argmin(solutions)
    worst_fitness = solutions[worst_idx]
    worst_solution = values[worst_idx]

    mean_iters = np.mean(iterations_number)
    best_iters = np.min(iterations_number)
    worst_iters = np.max(iterations_number)

    print(f"\n\nHiperparametry algorytmu: {solver.get_parameters()}")
    print(f"Solutions: {solutions}")
    print('Mean:', np.round(mean, 2))
    print('Standard deviation:',  np.round(std, 2))
    print(f'Mean iterations: {mean_iters}, Best iterations: {best_iters}, Worst iterations: {worst_iters}')
    print("\nBest solution overall:",  np.round(best_fitness, 2))
    print("Best fitness overall:", best_solution)
    print("\nWorst solution overall:", np.round(worst_fitness, 2))
    print("Worst fitness overall:", worst_solution)

    # Zbieżność
    fig = plt.figure(figsize=(10, 8))
    fig.subplots_adjust(left=0.07, right=0.9, bottom=0.07, top=0.9)
    for i, history in enumerate(fitness_history):
        plt.plot(history, label=f'Uruchomienie {i + 1}')
    plt.xlabel("Iteracja")
    plt.ylabel("Wartość funkcji celu")
    plt.title(f"Wykres 1. Zbieżność algorytmu genetycznego dla parametrów AG: {solver.get_parameters()}.")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Rozrzut
    fig = plt.figure(figsize=(10, 8))
    fig.subplots_adjust(left=0.07, right=0.9, bottom=0.07, top=0.9)
    plt.scatter(range(len(solutions)), solutions)
    plt.xlabel("Numer uruchomienia")
    plt.ylabel("Najlepsze wartości funkcji celu")
    plt.title(f"Wykres 2. Rozrzut wyników dla parametrów AG: {solver.get_parameters()}.")
    plt.grid(True)
    plt.show()
