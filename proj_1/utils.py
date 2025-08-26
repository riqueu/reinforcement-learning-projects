import numpy as np
import random
import pickle

class State:
    def __init__(self, battery_level=2):
        # 1 para bateria baixa e 2 para bateria alta
        self.battery_level = battery_level
    
    def hash(self):
        return self.battery_level
    
    def set_battery(self, battery_level):
        self.battery_level = battery_level


class Environment:
    def __init__(self, α, β, r_search, r_wait, robot):
        self.α = α
        self.β = β
        self.r_search = r_search
        self.r_wait = r_wait
        self.robot = robot
        self.robot.set_state(2)

    def step(self, state, action):
        next_state, reward = state, None

        # Bateria Baixa
        if state.battery_level == 1:
            # Procurou com bateria baixa
            if action == "search":
                if random.random() < self.β:
                    reward = self.r_search
                else:
                    next_state = State(2)
                    reward = -3
            
            # Esperou com bateria baixa
            elif action == "wait":
                reward = self.r_wait

            # Recarregou com bateria baixa
            elif action == "recharge":
                next_state = State(2)
                reward = 0

        # Bateria Alta
        if state.battery_level == 2:
            # Procurou com bateria alta
            if action == "search":
                if random.random() >= self.α:
                    # Bateria vai para o low
                    next_state = State(1)
                reward = self.r_search

            # Esperou com bateria alta
            elif action == "wait":
                reward = self.r_wait
        
        self.robot.set_state(next_state.hash())
        self.robot.set_reward(reward)


class Robot:
    def __init__(self, ε=0.1, lr=0.1):
        self.ε = ε
        self.lr = lr
        self.estimations = np.ndarray.ones(shape=(2, 3), dtype=object)
        self.estimations[1, 2] = 0
        self.actions_list = ["search", "wait", "recharge"]
        self.state_hist = []
        self.reward_hist = []
        self.action_hist = []

    def reset(self):
        self.state_hist = []
        self.set_state(2)
        self.reward_hist = []
        self.action_hist = []
    
    def set_state(self, state):
        self.state_hist.append(state)

    def set_reward(self, reward):
        self.reward_hist.append(reward)

    def act(self):
        state = self.state_hist[-1]
        values = []
        
        # caso epsilon
        if random.random() < self.ε:
            if state.battery_level == 2:
                action = random.choice(self.actions_list[:-1])
            else:
                action = random.choice(self.actions_list)

            self.action_hist.append(self.actions_list.index(action))
            return action
        
        # lida com ação invalida (recarregar com bateria cheia)
        if state.battery_level == 2:
            for action_idx in range(self.estimations.shape[1] - 1):
                values.append((self.estimations[state-1][action_idx], action_idx))
        else:
            for action_idx in range(self.estimations.shape[1]):
                values.append((self.estimations[state-1][action_idx], action_idx))
        
        np.random.shuffle(values)
        values.sort(key=lambda x: x[0], reverse=True)
        action_idx = values[0][1]

        return self.actions_list[action_idx]

    def backup(self):
        for k in range(len(self.state_hist) - 1):
            td_error = self.reward_hist[k] + self.estimations[self.state_hist[k+1]][self.action_hist[k]] - self.estimations[self.state_hist[k]][self.action_hist[k]]
            self.estimations[self.state_hist[k+1]][self.action_hist[k]] += self.lr * td_error

    def save_policy(self):
        with open('policy.pkl', 'wb') as f:
            pickle.dump(self.estimations, f)

    def load_policy(self):
        with open('policy.pkl', 'rb') as f:
            self.estimations = pickle.load(f)
