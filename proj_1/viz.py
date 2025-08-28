import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def save_fig_rewards(rewards):
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=rewards)
    plt.title("Training Rewards Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Total Reward")
    plt.tight_layout()
    plt.savefig("rewards.png")
    plt.close()


def save_fig_action_distribution(action_count):
    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(action_count.keys()), y=list(action_count.values()))
    plt.title("Action Distribution")
    plt.xlabel("Action")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("action_distribution.png")
    plt.close()
    
    
# softmax para converter para probabilidades
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)
    
    
def save_fig_optimal_policy_heatmap(optimal_policy):
    plt.figure(figsize=(8, 6))

    # labels
    action_names = ["Search", "Wait", "Recharge"]
    state_names = ["Low Battery", "High Battery"]
    
    
    # matriz da politica
    policy_matrix = np.zeros((2, 3))
    policy_matrix[0, :] = softmax(optimal_policy[0, :3])
    policy_matrix[1, :2] = softmax(optimal_policy[1, :2])
    policy_matrix[1, 2] = np.nan # recarregar com bateria cheia
    
    # Plot heatmap with probabilities
    sns.heatmap(policy_matrix, cmap="Blues", 
                xticklabels=action_names, yticklabels=state_names,
                annot=True, fmt='.3f', cbar=True)
    
    plt.title("Optimal Policy Probabilities")
    plt.xlabel("Actions")
    plt.ylabel("Battery State")
    plt.tight_layout()
    plt.savefig("optimal_policy_heatmap.png")
    plt.close()
