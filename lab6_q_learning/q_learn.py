"""
Author: Viktoriia Nowotka
"""
import gymnasium as gym
from minigrid.minigrid_env import MiniGridEnv
import numpy as np

# Q-learning z epizodami oraz ϵ-zachłanną strategią wyboru akcji
# * Dostosuj rozwiązanie, aby działało również dla losowych pozycji agenta i celu.

class QLearning:
    def __init__(self, actions, n_dirs, e_max, max_steps=100, goal_pos=(2, 2), seed=42):
        self.Q = np.array([])
        self.seed = seed
        self.e_max = e_max

        self.env = gym.make("MiniGrid-FourRooms-v0", max_steps=max_steps, goal_pos=goal_pos, render_mode="human")
        self.base_env = self.env.unwrapped

        self.U = actions                    # actions: left, right, forward
        self.n_dirs = n_dirs                # 4 directions
        self.X = self.__init_states()       # available states: (x, y, dir)

        self.state_to_id = self.__build_state_mapping()

    def __valid_positions(self):
        valid = []

        for x in range(self.base_env.width):
            for y in range(self.base_env.height):
                cell = self.base_env.grid.get(x, y)
                if cell is None or cell.can_overlap():
                    valid.append((x, y))
        return valid

    def __init_states(self):
        return self.__valid_positions()

    def __build_state_mapping(self):
        # state_to_id: (x, y, dir) -> index
        state_to_id = {}
        idx = 0

        for (x, y) in self.__valid_positions():
            for d in range(4):
                state_to_id[(x, y, d)] = idx
                idx += 1

        return state_to_id

    def __define_x(self):
        x, y = self.base_env.agent_pos
        d = self.base_env.agent_dir
        return (x, y, d), self.state_to_id[(x, y, d)]

    def __set_random_agent_pos(self):
        valid = self.__valid_positions()
        pos = valid[np.random.randint(len(valid))]
        self.base_env.agent_pos = np.array(pos)
        self.base_env.agent_dir = np.random.randint(4)

    @staticmethod
    def print_agent_position(pos):
        x, y, dir = pos
        print(f"Agent grid pos: {x}; {y} | dir: {dir}")

    @staticmethod
    def choose_action(x, Q, eps):
        if np.random.random() < eps:
            return np.random.choice(Q.shape[1])  # eksploracja: losowa akcja

        q_values = Q[x]
        max_q = np.max(q_values)
        best_actions = np.where(q_values == max_q)[0]

        return np.random.choice(best_actions)    # eksploatacja: najlepsze

    @staticmethod
    def log(e, t):
        if t % 100 == 0:
            print(f"Episode: {e+1}, step: {t}")

    def train(self, b, y, eps=0.1):
        e = 0
        Q = np.zeros((len(self.X) * self.n_dirs, len(self.U)))

        while e < self.e_max:
            self.env.reset(seed=self.seed)
            t = 0
            terminated = False
            truncated = False

            # kady epizod ma rózny punkt A
            self.__set_random_agent_pos()
            pos_x, x = self.__define_x()
            self.print_agent_position(pos_x)

            while not terminated or not truncated:
                self.log(e, t)
                x_prev = x

                act = self.choose_action(x, Q, eps)
                _, rew, terminated, truncated, _ = self.env.step(act)
                _, x = self.__define_x()

                delta = rew + y * max(Q[x]) - Q[x_prev][act]
                Q[x][act] += b * delta

                t += 1

                if truncated:
                    print("STOP: timeout")
                    self.env.reset(seed=self.seed)

                if terminated:
                    print("STOP: reached goal")
                    self.env.reset(seed=self.seed)
            e += 1

        self.Q = Q

    def run(self, agent_pos=(7,7)):
        self.env.reset(seed=self.seed)
        x = self.__define_x()

        terminated = False
        truncated = False
        n_steps = 0
        eps = 0

        while not terminated and not truncated:
            action = self.choose_action(x, self.Q, eps)
            _, _, terminated, truncated, _ = self.env.step(action)

            x = self.__define_x()
            n_steps += 1

        print(f"Run finished after {n_steps} steps")
        return n_steps

    def close(self):
        self.env.close()