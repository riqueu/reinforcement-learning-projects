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
        if state == 1:
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
        if state == 2:
            # Procurou com bateria alta
            if action == "search":
                if random.random() >= self.α:
                    # Bateria vai para o low
                    next_state = State(1)
                reward = self.r_search

            # Esperou com bateria alta
            elif action == "wait":
                reward = self.r_wait
        
        self.robot.set_state(next_state)
        self.robot.set_reward(reward)


class Robot:
    def __init__(self, ε=0.001, lr=0.1):
        self.ε = ε
        self.lr = lr
        self.estimations = np.ones(shape=(2, 3))
        self.estimations[1, 2] = 0
        self.actions_list = ["search", "wait", "recharge"]
        self.state_hist = [] # histórico de baterias (int, 1 para low, 2 para high)
        self.reward_hist = [] # histórico de recompensas (float)
        self.action_hist = [] # histórico de ação (int, index)

    def reset(self):
        self.state_hist = []
        self.set_state(2)
        self.reward_hist = []
        self.action_hist = []
    
    def set_state(self, state):
        # garante que está adicionando a bateria, e não o estado
        if isinstance(state, State):
            self.state_hist.append(state.hash())
        else:
            self.state_hist.append(state)

    def set_reward(self, reward):
        self.reward_hist.append(reward)

    def act(self):
        state = self.state_hist[-1]
        state_idx = state - 1
        values = []
        
        # caso epsilon
        if random.random() < self.ε:
            if state == 2:
                action = random.choice(self.actions_list[:-1])
            else:
                action = random.choice(self.actions_list)

            self.action_hist.append(self.actions_list.index(action))
            return action
        
        # lida com ação invalida (recarregar com bateria cheia)
        if state == 2:
            for action_idx in range(self.estimations.shape[1] - 1):
                values.append((self.estimations[state_idx][action_idx], action_idx))
        else:
            for action_idx in range(self.estimations.shape[1]):
                values.append((self.estimations[state_idx][action_idx], action_idx))
        
        np.random.shuffle(values)
        values.sort(key=lambda x: x[0], reverse=True)
        action_idx = values[0][1]

        self.action_hist.append(action_idx)
        return self.actions_list[action_idx]

    def backup(self):
        for k in range(len(self.state_hist) - 1):
            curr_state_idx = self.state_hist[k] - 1
            next_state_idx = self.state_hist[k+1] - 1
            action_idx = self.action_hist[k]
            reward = self.reward_hist[k]
            #gamma = 0.95
            
            # encontra Q-value máximo em cada caso possível
            if self.state_hist[k+1] == 2:
                next_values = [self.estimations[next_state_idx][0], 
                            self.estimations[next_state_idx][1]]
            else:
                next_values = [self.estimations[next_state_idx][a] for a in range(3)]
                
            max_next_q = max(next_values)
            
            # update td com max_q
            curr_q = self.estimations[curr_state_idx][action_idx]
            td_error = reward + max_next_q - curr_q
            self.estimations[curr_state_idx][action_idx] += self.lr * td_error

    def save_policy(self):
        with open('policy.pkl', 'wb') as f:
            pickle.dump(self.estimations, f)

    def load_policy(self):
        with open('policy.pkl', 'rb') as f:
            self.estimations = pickle.load(f)
