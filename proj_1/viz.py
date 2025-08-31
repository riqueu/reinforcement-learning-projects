import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def save_fig_rewards(rewards: list[float]) -> None:
    """Plot and save the total reward per epoch
    
    Args:
        rewards (list[float]): List of rewards for each epoch.
    """
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=rewards)
    plt.title("Training Rewards Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Total Reward")
    plt.tight_layout()
    plt.savefig("rewards.png")
    plt.close()


def save_fig_multiple_rewards(all_rewards: np.ndarray, avg_rewards: np.ndarray, std_rewards: np.ndarray) -> None:
    """plot reward curves for multiple training runs
    
    Args:
        all_rewards (np.ndarray): rewards from all runs.
        avg_rewards (np.ndarray): avg rewards per epoch.
        std_rewards (np.ndarray): std of rewards per epoch.
    """
    plt.figure(figsize=(10, 6))
    # runs individuais em baixa opacidade
    colors = plt.cm.viridis(np.linspace(0, 1, len(all_rewards)))
    for i, rewards in enumerate(all_rewards):
        plt.plot(rewards, alpha=0.15, color=colors[i])
    # média com intervalo de confiança
    plt.plot(avg_rewards, linewidth=2, color='red', label="Average")
    plt.fill_between(range(len(avg_rewards)), 
                    avg_rewards - std_rewards, 
                    avg_rewards + std_rewards, 
                    alpha=0.2, color='red')
    # título e rótulos do plot
    plt.title("Training Rewards Over Multiple Runs")
    plt.xlabel("Epoch")
    plt.ylabel("Total Reward")
    plt.legend(loc='lower right')
    #plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("rewards_multiple_runs.png")
    plt.close()


def save_fig_action_distribution(action_count) -> None:
    """Plot distribution of actions taken by the agent

    Args:
        action_count (dict[str, int]): dict with action counts.
    """
    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(action_count.keys()), y=list(action_count.values()))
    plt.title("Action Distribution")
    plt.xlabel("Action")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("action_distribution.png")
    plt.close()
    
    
# softmax para converter para probabilidades
def softmax(x: np.ndarray) -> np.ndarray:
    """Softmax of a numpy array

    Args:
        x (np.ndarray): Input array.
    Returns:
        np.ndarray: Softmax probabilities.
    """
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


def save_fig_optimal_policy_heatmap(optimal_policy) -> None:
    """Heatmap of optimal policy probabilities for each state-action pair

    Args:
        optimal_policy (np.ndarray): Q-value table (2x3) for the agent.
    """
    plt.figure(figsize=(8, 6))
    # labels
    action_names = ["Search", "Wait", "Recharge"]
    state_names = ["Low Battery", "High Battery"]
    # matriz da politica
    # (utilizamos softmax para converter Q-values em probabilidades)
    policy_matrix = np.zeros((2, 3))
    policy_matrix[0, :] = softmax(optimal_policy[0, :3])
    policy_matrix[1, :2] = softmax(optimal_policy[1, :2])
    policy_matrix[1, 2] = np.nan # recarregar com bateria cheia (ação nula)
    # heatmap com probabilidades
    sns.heatmap(policy_matrix, cmap="Blues", 
                xticklabels=action_names, yticklabels=state_names,
                annot=True, fmt='.3f', cbar=True)
    plt.title("Optimal Policy Probabilities")
    plt.xlabel("Actions")
    plt.ylabel("Battery State")
    plt.tight_layout()
    plt.savefig("optimal_policy_heatmap.png")
    plt.close()