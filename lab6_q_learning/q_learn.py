"""
Author: Viktoriia Nowotka
"""
import gymnasium as gym
from minigrid.minigrid_env import MiniGridEnv
import numpy as np
from lab6_q_learning.solver import Solver

# Q-learning z epizodami oraz ϵ-zachłanną strategią wyboru akcji
# * Dostosuj rozwiązanie, aby działało również dla losowych pozycji agenta i celu.

class QLearning(Solver):
    def __init__(self, actions, n_dirs, e_max, max_steps=100, goal_pos=(2, 2), seed=42):
        self.seed = seed
        self.e_max = e_max
        self.max_steps = max_steps

        self.env = gym.make("MiniGrid-FourRooms-v0", max_steps=max_steps, goal_pos=goal_pos, render_mode="human")
        self.base_env = self.env.unwrapped
        self.env.reset(seed=self.seed)

        self.U = actions                    # actions: left, right, forward
        self.n_dirs = n_dirs                # 4 directions
        self.X = self.__init_states()       # available states: (x, y)

        self.state_to_id = self.__build_state_mapping()

    def __init_states(self):
        valid = []

        for x in range(self.base_env.width):
            for y in range(self.base_env.height):
                cell = self.base_env.grid.get(x, y)
                if cell is None or cell.can_overlap():
                    valid.append((x, y))
        return valid

    def __build_state_mapping(self):
        # state_to_id: (x, y, dir) -> index
        state_to_id = {}
        idx = 0

        for (x, y) in self.X:
            for d in range(4):
                state_to_id[(x, y, d)] = idx
                idx += 1

        return state_to_id

    def __define_x(self):
        x, y = self.base_env.agent_pos
        d = self.base_env.agent_dir

        key = (x, y, d)
        if key not in self.state_to_id:
            raise ValueError(f"State {key} not in state_to_id")

        return (x, y, d), self.state_to_id[key]

    def __set_random_agent_pos(self):
        x, y = self.X[np.random.randint(len(self.X))]

        self.base_env.agent_pos = np.array([x, y])
        self.base_env.agent_dir = np.random.randint(self.n_dirs)

    def __set_agent_pos(self, pos):
        self.base_env.agent_pos = np.array(pos)
        self.base_env.agent_dir = 0

    @staticmethod
    def print_agent_position(pos):
        x, y, d = pos
        print(f"Agent grid pos: {x}; {y} | dir: {d}")

    @staticmethod
    def choose_action(ind_x, Q, eps):
        if np.random.random() < eps:
            rand_act = np.random.choice(Q.shape[1])
            return rand_act  # eksploracja: losowa akcja

        q_values = Q[ind_x]
        max_q = np.max(q_values)
        best_actions = np.where(q_values == max_q)[0]
        return np.random.choice(best_actions)    # eksploatacja: najlepsza z możliwych akcji

    @staticmethod
    def log(e, t):
        if t % 100 == 0:
            print(f"Episode: {e+1}, step: {t}")

    def get_parameters(self):
        return f"episodes={self.e_max}, max_steps={self.max_steps} b={self.b}, y={self.y}, eps={self.eps}"

    def train(self, b, y, eps=0.1):
        self.b = b
        self.y = y
        self.eps = eps
        Q = np.zeros((len(self.state_to_id), len(self.U)))

        for e in range(self.e_max):
            self.env.reset(seed=self.seed)

            t = 0
            terminated = False
            truncated = False

            # każdy epizod ma różny punkt A
            # self.__set_random_agent_pos()
            self.__set_agent_pos(pos=(7,7))
            pos_x, ind_x = self.__define_x()
            self.print_agent_position(pos_x)

            while not terminated and not truncated:
                self.log(e, t)

                act = self.choose_action(ind_x, Q, eps)
                _, rew, terminated, truncated, _ = self.env.step(act)
                _, ind_x_next = self.__define_x()

                delta = rew + y * max(Q[ind_x_next]) - Q[ind_x][act]
                Q[ind_x][act] += b * delta

                ind_x = ind_x_next
                t += 1

                if truncated:
                    print("STOP: timeout\n")
                    break

                if terminated:
                    print("STOP: reached goal\n")
                    break

        self.Q = Q

    def run(self, agent_pos=(7,7)):
        self.env.reset(seed=self.seed)
        self.__set_agent_pos(agent_pos)
        _, ind_x = self.__define_x()

        terminated = False
        truncated = False
        n_steps = 0
        eps = 0

        while not terminated and not truncated:
            action = self.choose_action(ind_x, self.Q, eps)
            _, _, terminated, truncated, _ = self.env.step(action)

            _, ind_x = self.__define_x()
            n_steps += 1

        return n_steps

    def close(self):
        self.env.close()
