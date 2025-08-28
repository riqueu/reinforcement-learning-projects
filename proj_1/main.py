from utils import *
from viz import *
import numpy as np

# probabilities
α = 0.3
β = 0.2

# rewards
r_search = 3.5
r_wait = 0.5
assert r_search > r_wait

# training parameters
epochs = 1000
steps = 1000

def train(epochs=epochs, steps=steps):
    robot = Robot()
    env = Environment(α, β, r_search, r_wait, robot)
    rewards = []
    action_list = ["search", "wait", "recharge"]
    action_count = {action: 0 for action in action_list}

    for i in range(1, epochs + 1):
        for j in range(steps):
            action = robot.act()
            env.step(state=robot.state_hist[-1], action=action)
            action_count[action] += 1
            #if j % 100 == 0:
                #print(j, ": ", end="")
                #print(robot.state_hist[-2], action_list[robot.action_hist[-1]], robot.reward_hist[-1])
        robot.backup()
        if i % 50 == 0:
            print(f'Epoch: {i} | Reward: {np.sum(robot.reward_hist)}\r', end='')
        
        rewards.append(np.sum(robot.reward_hist))
        robot.reset()
        #print(action_count)
    robot.save_policy()
    print("")
    
    return rewards, action_count
          

def write_rewards(rewards):
    with open("rewards.txt", "w") as f:
        for reward in rewards:
            f.write(f"{reward}\n")

if __name__ == '__main__':
    # treina e salva histórico de recompensas
    rewards, action_count = train()
    # escreve arquivo rewards.txt
    write_rewards(rewards)
    save_fig_rewards(rewards)
    save_fig_action_distribution(action_count)