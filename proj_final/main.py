"""This tutorial shows how to train an MATD3 agent on the simple speaker listener multi-particle environment.

Authors: Michael (https://github.com/mikepratt1), Nickua (https://github.com/nicku-a)
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from pettingzoo.mpe import simple_speaker_listener_v4
from mpe2 import simple_speaker_listener_v4

from agilerl.algorithms import MATD3
from agilerl.algorithms.core.registry import HyperparameterConfig, RLParameter
from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.algorithms.core.registry import NetworkGroup
from agilerl.utils.utils import (
    create_population,
    default_progress_bar,
    make_multi_agent_vect_envs,
)

if __name__ == "__main__":
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("===== AgileRL Online Multi-Agent Demo =====")

    # Define the network configuration
    # Deeper, more stable architecture for better feature learning
    # The commented config above is agent-specific but not supported by create_population
    # Note: latent_dim must be <= observation space dimensions
    NET_CONFIG = {
        "latent_dim": 128,
        "encoder_config": {"hidden_size": [256, 128]},
        "head_config": {"hidden_size": [128, 64]},
    }

    # Define the initial hyperparameters
    INIT_HP = {
        "POPULATION_SIZE": 5,
        "ALGO": "MATD3",
        "BATCH_SIZE": 256,
        "O_U_NOISE": True,
        "EXPL_NOISE": 0.15,
        "MEAN_NOISE": 0.0,
        "THETA": 0.15,
        "DT": 0.01,
        "LR_ACTOR": 5e-5,
        "LR_CRITIC": 1e-3,
        "GAMMA": 0.99,
        "MEMORY_SIZE": 250000,
        "LEARN_STEP": 8,
        "TAU": 0.005,
        "POLICY_FREQ": 4,
        "MAX_GRAD_NORM": 10.0,
    }

    num_envs = 8

    def make_env():
        return simple_speaker_listener_v4.parallel_env(continuous_actions=True)

    env = make_multi_agent_vect_envs(env=make_env, num_envs=num_envs)

    # Configure the multi-agent algo input arguments
    observation_spaces = [env.single_observation_space(agent) for agent in env.agents]
    action_spaces = [env.single_action_space(agent) for agent in env.agents]

    # Append number of agents and agent IDs to the initial hyperparameter dictionary
    INIT_HP["AGENT_IDS"] = env.agents

    # Mutation config for RL hyperparameters
    hp_config = HyperparameterConfig(
        lr_actor=RLParameter(min=1e-5, max=5e-4),
        lr_critic=RLParameter(min=5e-4, max=3e-3),
        batch_size=RLParameter(min=128, max=512, dtype=int),
        learn_step=RLParameter(min=4, max=32, dtype=int, grow_factor=1.2, shrink_factor=0.85),
        tau=RLParameter(min=0.003, max=0.01),
        policy_freq=RLParameter(min=3, max=5, dtype=int),
        gamma=RLParameter(min=0.97, max=0.995)
    )

    # Create a population ready for evolutionary hyper-parameter optimisation
    pop = create_population(
        "MATD3",
        observation_spaces,
        action_spaces,
        NET_CONFIG,
        INIT_HP,
        hp_config=hp_config,
        population_size=INIT_HP["POPULATION_SIZE"],
        num_envs=num_envs,
        device=device,
    )

    # Configure the multi-agent replay buffer
    field_names = ["obs", "action", "reward", "next_obs", "done"]
    memory = MultiAgentReplayBuffer(
        INIT_HP["MEMORY_SIZE"],
        field_names=field_names,
        agent_ids=INIT_HP["AGENT_IDS"],
        device=device,
    )

    # Instantiate a tournament selection object (used for HPO)
    tournament = TournamentSelection(
        tournament_size=2,  # Tournament selection size
        elitism=True,  # Elitism in tournament selection
        population_size=INIT_HP["POPULATION_SIZE"],  # Population size
        eval_loop=2,  # Evaluate using last N fitness scores
    )

    # Instantiate a mutations object (used for HPO)
    mutations = Mutations(
        no_mutation=0.50,
        architecture=0.02,
        new_layer_prob=0.01,
        parameters=0.25,
        activation=0.03,
        rl_hp=0.20,
        mutation_sd=0.06,
        rand_seed=42,
        device=device,
    )

    # Define training loop parameters
    max_steps = 2_000_000  # Max steps (default: 2000000)
    # max_steps = 400_000  # Max steps (default: 2000000)
    learning_delay = 1000  # Steps before starting learning
    evo_steps = 10_000  # Evolution frequency
    eval_steps = None  # Evaluation steps per episode - go until done
    eval_loop = 2  # Number of evaluation episodes
    elite = pop[0]  # Assign a placeholder "elite" agent
    total_steps = 0
    noise_start = INIT_HP["EXPL_NOISE"]
    noise_decay = 400000  # gradual exploration reduction
    noise_end = 0.005  # maintain minimal exploration

    # Lista para armazenar pontuações médias para plotagem
    training_scores_history = []

    # best agent hist
    best_fitness_history = []

    # TRAINING LOOP
    print("Training...")
    pbar = default_progress_bar(max_steps)
    while np.less([agent.steps[-1] for agent in pop], max_steps).all():
        pop_episode_scores = []
        for agent in pop:  # Loop through population
            agent.set_training_mode(True)
            obs, info = env.reset()  # Reset environment at start of episode
            scores = np.zeros(num_envs)
            completed_episode_scores = []
            steps = 0
            for idx_step in range(evo_steps // num_envs):
                action, raw_action = agent.get_action(
                    obs=obs, infos=info
                )  # Predict action
                next_obs, reward, termination, truncation, info = env.step(
                    action
                )  # Act in environment

                # --- NOISE DECAY ---
                decay_progress = min(total_steps / noise_decay, 1.0)
                explNoise = noise_start + decay_progress * (noise_end - noise_start)
                agent.EXPL_NOISE = explNoise

                scores += np.sum(np.array(list(reward.values())).transpose(), axis=-1)
                total_steps += num_envs
                steps += num_envs

                # Save experiences to replay buffer
                memory.save_to_memory(
                    obs,
                    raw_action,
                    reward,
                    next_obs,
                    termination,
                    is_vectorised=True,
                )

                # Learn according to learning frequency
                # Handle learn steps > num_envs
                if agent.learn_step > num_envs:
                    learn_step = agent.learn_step // num_envs
                    if (
                        idx_step % learn_step == 0
                        and len(memory) >= agent.batch_size
                        and memory.counter > learning_delay
                    ):
                        experiences = memory.sample(
                            agent.batch_size
                        )  # Sample replay buffer
                        agent.learn(
                            experiences
                        )  # Learn according to agent's RL algorithm

                # Handle num_envs > learn step; learn multiple times per step in env
                elif (
                    len(memory) >= agent.batch_size and memory.counter > learning_delay
                ):
                    for _ in range(num_envs // agent.learn_step):
                        experiences = memory.sample(
                            agent.batch_size
                        )  # Sample replay buffer
                        agent.learn(
                            experiences
                        )  # Learn according to agent's RL algorithm

                obs = next_obs

                # Calculate scores and reset noise for finished episodes
                reset_noise_indices = []
                term_array = np.array(list(termination.values())).transpose()
                trunc_array = np.array(list(truncation.values())).transpose()
                for idx, (d, t) in enumerate(zip(term_array, trunc_array)):
                    if np.any(d) or np.any(t):
                        completed_episode_scores.append(scores[idx])
                        agent.scores.append(scores[idx])
                        scores[idx] = 0
                        reset_noise_indices.append(idx)

                agent.reset_action_noise(reset_noise_indices)

            pbar.update(evo_steps // len(pop))

            agent.steps[-1] += steps
            pop_episode_scores.append(completed_episode_scores)

        # Evaluate population
        fitnesses = [
            agent.test(
                env,
                max_steps=eval_steps,
                loop=eval_loop,
            )
            for agent in pop
        ]
        mean_scores = [
            (
                np.mean(episode_scores)
                if len(episode_scores) > 0
                else 0
            )
            for episode_scores in pop_episode_scores
        ]
        
        # Salvar pontuação média da população para plotagem
        population_mean_score = np.mean([score for score in mean_scores if isinstance(score, (int, float))])
        training_scores_history.append(population_mean_score)
        best_fitness_history.append(max(fitnesses))

        mean_scores_display = [
            (
                score if isinstance(score, (int, float))
                else "0 completed episodes"
            )
            for score in mean_scores
        ]

        pbar.write(
            f"--- Global steps {total_steps} ---\n"
            f"Steps {[agent.steps[-1] for agent in pop]}\n"
            f"Scores: {mean_scores_display}\n"
            f"Fitnesses: {['%.2f' % fitness for fitness in fitnesses]}\n"
            f"5 fitness avgs: {['%.2f' % np.mean(agent.fitness[-5:]) for agent in pop]}\n"
            f"Mutations: {[agent.mut for agent in pop]}"
        )

        # Tournament selection and population mutation
        elite, pop = tournament.select(pop)
        
        # Strong elitism: preserve the best agent to prevent catastrophic mutations
        elite_backup = elite
        
        pop = mutations.mutation(pop)
        
        # Restore elite (never mutate the best agent)
        pop[0] = elite_backup

        # Update step counter
        for agent in pop:
            agent.steps.append(agent.steps[-1])

    # Save the trained algorithm
    path = "./models/MATD3"
    filename = "MATD3_trained_agent.pt"
    os.makedirs(path, exist_ok=True)
    save_path = os.path.join(path, filename)
    elite.save_checkpoint(save_path)
    
    # Plotar e salvar a evolução das pontuações
    plt.figure(figsize=(12, 6))
    plt.plot(training_scores_history, linewidth=2)
    plt.title('Evolução das Pontuações Médias Durante o Treinamento', fontsize=14)
    plt.xlabel('Iterações de Evolução', fontsize=12)
    plt.ylabel('Pontuação Média da População', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Salvar o gráfico
    plot_path = os.path.join(path, "training_scores_evolution.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Gráfico de evolução das pontuações salvo em: {plot_path}")
    
    # Salvar dados das pontuações em arquivo numpy
    scores_data_path = os.path.join(path, "training_scores_history.npy")
    np.save(scores_data_path, np.array(training_scores_history))
    print(f"Dados das pontuações salvos em: {scores_data_path}")
    
    plt.show()

    pbar.close()
    env.close()
