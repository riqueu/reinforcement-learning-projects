import seaborn as sns
import matplotlib.pyplot as plt


def save_fig_rewards(rewards):
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=rewards)
    plt.title("Training Rewards Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Total Reward")
    plt.savefig("rewards.png")
    plt.close()


def save_fig_action_distribution(action_count):
    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(action_count.keys()), y=list(action_count.values()))
    plt.title("Action Distribution")
    plt.xlabel("Action")
    plt.ylabel("Count")
    plt.savefig("action_distribution.png")
    plt.close()