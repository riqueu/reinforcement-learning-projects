import numpy as np
import random
import pickle
from typing import Union

class State:
    def __init__(self, battery_level: int = 2) -> None:
        """Initializes the state representing the robot's battery level

        Args:
            battery_level (int): 1 for low, 2 for high.
        """
        # 1 para bateria baixa e 2 para bateria alta
        self.battery_level = battery_level

    def hash(self) -> int:
        """Unique hash for the state (just the battery level in this problem's case)
        
        Returns:
            int: Battery level (1 or 2).
        """
        return self.battery_level

    def set_battery(self, battery_level: int) -> None:
        """Set the battery level for this state

        Args:
            battery_level (int): New battery level.
        """
        self.battery_level = battery_level


class Environment:
    def __init__(self, α: float, β: float, r_search: float, r_wait: float, robot: 'Robot') -> None:
        """Env for recycling robot mdp.

        Args:
            α (float): prob of staying high after search.
            β (float): prob of staying low after search.
            r_search (float): reward for searching.
            r_wait (float): reward for waiting.
            robot (Robot): agent interacting with the environment.
        """
        self.α = α
        self.β = β
        self.r_search = r_search
        self.r_wait = r_wait
        self.robot = robot
        self.robot.set_state(2)

    def step(self, state: int, action: str) -> None:
        """Execute one step in the env given current state and action

        Args:
            state (int): battery state (1 or 2).
            action (str): Action taken by the agent.
        """
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
    def __init__(self, ε: float = 0.001, lr: float = 0.1) -> None:
        """RL agent for recycling robot

        Args:
            ε (float): Epsilon for epsilon-greedy policy.
            lr (float): Learning rate for TD updates.
        """
        self.ε = ε
        self.lr = lr
        self.estimations = np.ones(shape=(2, 3))
        self.estimations[1, 2] = 0
        self.actions_list = ["search", "wait", "recharge"]
        self.state_hist: list[int] = [] # histórico de baterias (int, 1 para low, 2 para high)
        self.reward_hist: list[float] = [] # histórico de recompensas (float)
        self.action_hist: list[int] = [] # histórico de ação (int, index)

    def reset(self) -> None:
        """Reset the agent's history for a new episode.
        """
        self.state_hist = []
        self.set_state(2)
        self.reward_hist = []
        self.action_hist = []
    
    def set_state(self, state: Union[int, 'State']) -> None:
        """Add current state to agent's battery
        
        Args:
            state (int or State): Battery state or State object.
        """
        # garante que está adicionando a bateria, e não o estado
        if isinstance(state, State):
            self.state_hist.append(state.hash())
        else:
            self.state_hist.append(state)

    def set_reward(self, reward: float) -> None:
        """Add received reward to the agent's history

        Args:
            reward (float): Reward received.
        """
        self.reward_hist.append(reward)

    def act(self) -> str:
        """Select action via epsilon-greedy policy

        Returns:
            str: Action chosen ("search", "wait", or "recharge").
        """
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

        # embaralha e ordena as ações
        np.random.shuffle(values)
        values.sort(key=lambda x: x[0], reverse=True)
        action_idx = values[0][1]
        self.action_hist.append(action_idx)
        
        return self.actions_list[action_idx]

    def backup(self) -> None:
        """Update Q-val estimations using TD algorithm.
        """
        for k in range(len(self.state_hist) - 1):
            curr_state_idx = self.state_hist[k] - 1
            next_state_idx = self.state_hist[k+1] - 1
            action_idx = self.action_hist[k]
            reward = self.reward_hist[k]
            
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

    def save_policy(self) -> None:
        """Save learned Q-value table to a pkl file
        """
        with open('policy.pkl', 'wb') as f:
            pickle.dump(self.estimations, f)

    def load_policy(self) -> None:
        """Load Q-value table from a pkl file
        """
        with open('policy.pkl', 'rb') as f:
            self.estimations = pickle.load(f)
