from utils import *
from viz import *
import numpy as np


def train(epochs: int, steps: int, alpha: float, beta: float, r_s: float, r_w: float, save: bool = False) -> tuple[list[float], dict[str, int], np.ndarray]:
    """runs training session for the recycling robot RL agent
    
    Args:
        epochs (int): number of epochs.
        steps (int): steps per epoch.
        alpha (float): probability of battery staying high after searching.
        beta (float): probability of battery staying low after searching.
        r_s (float): reward for searching.
        r_w (float): reward for waiting.

    Returns:
        rewards (list[float]): total rewards per epoch.
        action_count (dict[str, int]): count of each action taken.
        optimal_policy (np.ndarray): learned Q-value table.
    """
    robot = Robot()
    env = Environment(alpha, beta, r_s, r_w, robot)
    rewards: list[float] = []
    action_list: list[str] = ["search", "wait", "recharge"]
    action_count: dict[str, int] = {action: 0 for action in action_list}

    for i in range(1, epochs + 1):
        for j in range(steps):
            # Escolhe ação com base na política epsilon-greedy e executa no ambiente
            action = robot.act()
            env.step(state=robot.state_hist[-1], action=action)
            action_count[action] += 1

            # backup a cada 200 passos
            if j % 200 == 0 and j > 0:
                robot.backup()
        
        # backup ao final de uma epoch
        robot.backup()

        if i % 50 == 0:
            print(f'Epoch: {i} | Reward: {np.sum(robot.reward_hist)}\r', end='')

        rewards.append(np.sum(robot.reward_hist))
        robot.reset()

    # Obtém a política ótima aprendida e a salva
    optimal_policy = robot.estimations
    robot.save_policy()
    print("")

    # Escreve rewards.txt após o treinamento principal se save for True
    if save:
        write_rewards(rewards, "rewards.txt")
    
    return rewards, action_count, optimal_policy


def train_multiple_runs(params: dict, num_runs: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Runs multiple independent training sessions for the recycling robot RL agent

    Args:
        num_runs (int): Number of independent training runs.
        params (dict): Dictionary of parameters to pass to the train function.

    Returns:
        avg_rewards (np.ndarray): avg rewards per epoch across runs.
        std_rewards (np.ndarray): std of rewards per epoch.
        all_rewards_array (np.ndarray): All rewards from all runs.
    """
    all_rewards: list[list[float]] = []
    
    # Treina multiplas vezes e salva as recompensas de cada treinamento
    for i in range(num_runs):
        print(f"Starting training run {i+1}/{num_runs}")
        rewards, _, _ = train(**params)
        all_rewards.append(rewards)
    
    # Calcula média e desvio padrão das recompensas
    all_rewards_array = np.array(all_rewards)
    avg_rewards = np.mean(all_rewards_array, axis=0)
    std_rewards = np.std(all_rewards_array, axis=0)
    
    return avg_rewards, std_rewards, all_rewards_array


def write_rewards(rewards: list[float], filename: str = "rewards.txt") -> None:
    """ Save a list of rewards in a text file, one per line

    Args:
        rewards (list[float]): List of rewards to save.
        filename (str): Output filename.
    """
    with open(filename, "w") as f:
        for reward in rewards:
            f.write(f"{reward}\n")


if __name__ == '__main__':
    # probabilidades
    α = 0.3
    β = 0.2

    # recompensas
    r_search = 3.5
    r_wait = 0.5
    assert r_search > r_wait # condição do enunciado

    # parâmetros para treinamento
    epochs = 1000
    steps = 1000

    params = {
        "epochs": epochs,
        "steps": steps,
        "alpha": α,
        "beta": β,
        "r_s": r_search,
        "r_w": r_wait
    }

    # treinamento principal e salva rewards.txt
    rewards, action_count, optimal_policy = train(**params, save=True)

    # plota resultados do treinamento principal
    save_fig_rewards(rewards)
    save_fig_action_distribution(action_count)
    save_fig_optimal_policy_heatmap(optimal_policy)

    # múltiplos treinamentos para suavizar o gráfico de recompensas por epoch
    num_runs = 10
    avg_rewards, std_rewards, all_rewards = train_multiple_runs(params, num_runs)
    save_fig_multiple_rewards(all_rewards, avg_rewards, std_rewards)