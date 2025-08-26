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
    def __init__(self, α, β, r_search, r_wait):
        self.α = α
        self.β = β
        self.r_search = r_search
        self.r_wait = r_wait
    
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
        
        # Returns: next_state, reward
        return next_state, reward

class Robot:
    def __init__(self, ε, lr):
        self.ε = ε
        self.lr = lr
        self.state_hist = []
        self.reward_hist = []
        self.action_hist = []

    def reset(self):
        self.state_hist = []
        self.reward_hist = []
        self.action_hist = []

    def act(self, state):
        #TODO: implementar policy para retornar ação (search, wait, ou recharge)
        pass

    def backup(self):
        #TODO: implementar backup da política (learning update com Temporal Diference)
        pass
    
    def save_policy(self):
        #TODO: salvar valores aprendidos
        pass

    def load_policy(self):
        #TODO: carregar valores aprendidos
        pass