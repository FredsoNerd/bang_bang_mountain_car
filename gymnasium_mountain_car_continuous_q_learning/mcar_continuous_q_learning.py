# reference: https://www.youtube.com/watch?v=_SWnNhM5w-g

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import pickle

def run(episodes = 100, train = True, render = True, epsilon_0 = 0.2, alpha = 0.9,
        gamma = 0.9, act_space_n = 20, obs_space_pos_n = 20, obs_space_vel_n = 20,
        q_table_file = None, rewards_plot_file = None, actions_hist_file = None):
    # initializes the environment
    env = gym.make('MountainCarContinuous-v0', render_mode = "human" if render else None)

    # space discretization
    space_pos = np.linspace(env.observation_space.low[0], env.observation_space.high[0], obs_space_pos_n)
    space_vel = np.linspace(env.observation_space.low[1], env.observation_space.high[1], obs_space_vel_n)
    low = env.action_space.low[0]
    high = env.action_space.high[0]
    space_act_to_cont = lambda x: low + (high-low)*x/(act_space_n-1)
    
    # initializes Q table
    if train:
        q_table = np.zeros((len(space_pos), len(space_vel), act_space_n))
    elif q_table_file:
        q_table = pickle.load(open(q_table_file, "br"))
    else:
        raise Exception("No trained table file was given!")

    # saves total rewards
    rewards = np.zeros((episodes,))
    actions = []

    # simulates the episodes
    for episode in tqdm.tqdm(range(episodes)):
        epsilon = epsilon_0/(episode+1)
        # resets environment
        observation, _ = env.reset()
        state_pos = np.digitize(observation[0], space_pos)
        state_vel = np.digitize(observation[1], space_vel)
        action = np.argmax(q_table[state_pos, state_vel, :])

        # single episode simulation
        while True:
            # epsilon-greedy action
            if np.random.random() < epsilon and train:
                action = np.random.randint(0, act_space_n)
            
            # runs single action step
            observation, reward, terminated, truncated, _ = env.step([space_act_to_cont(action)])
            new_state_pos = np.digitize(observation[0], space_pos)
            new_state_vel = np.digitize(observation[1], space_vel)
            new_action = np.argmax(q_table[new_state_pos, new_state_vel, :])
            
            # updates Q table values
            reward = -1
            Q_t0 = q_table[state_pos, state_vel, action]
            Q_t1 = q_table[new_state_pos, new_state_vel, new_action]
            q_table[state_pos, state_vel, action] += alpha*(reward + gamma*Q_t1 - Q_t0)

            # updates sistem states
            action = new_action
            state_pos = new_state_pos
            state_vel = new_state_vel
            rewards[episode] += reward
            actions.append(action)

            # ends if done simulation
            if terminated:
                break

    # saves trained table
    if q_table_file:
        with open(q_table_file, "wb") as f:
            pickle.dump(q_table, f)

    # saves rewards plot
    if rewards_plot_file:
        plt.plot(rewards)
        if not train:
            mean = rewards.mean()
            plt.axhline(mean, c='r')
        plt.savefig(rewards_plot_file)
        plt.close()
        
    # saves action choices histogram
    if actions_hist_file:
        plt.hist(actions)
        plt.savefig(actions_hist_file)
        plt.close()

# trains model and stores q-table and rewards
run(episodes=1000, train=True, render=False, act_space_n=20, epsilon_0=10,
    q_table_file = "mcar_continuous_q_table.pkl",
    rewards_plot_file = "mcar_continuous_learn_rewards_plot")

# generate actions histogram from trained table
run(episodes=200, train=False, render=False, act_space_n=20,
    q_table_file = "mcar_continuous_q_table.pkl",
    rewards_plot_file = "mcar_continuous_trained_rewards_plot",
    actions_hist_file = "mcar_continuous_trained_actions_hist")

# render running examples from trained table
run(episodes=5, train=False, render=True, act_space_n=20,
    q_table_file = "mcar_continuous_q_table.pkl")