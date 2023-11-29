# reference: https://www.youtube.com/watch?v=_SWnNhM5w-g
#reference: https://gist.github.com/botforge/64cbb71780e6208172bbf03cd9293553

import tqdm
import pickle
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def run(episodes = 100, obs_space_pos_n = 20, obs_space_vel_n = 20, q_table_file = None, gif_file = None):
    # initializes the environment
    env = gym.make('MountainCar-v0', render_mode = "rgb_array")

    # space discretization
    space_pos = np.linspace(env.observation_space.low[0], env.observation_space.high[0], obs_space_pos_n)
    space_vel = np.linspace(env.observation_space.low[1], env.observation_space.high[1], obs_space_vel_n)
    
    # initializes Q table
    q_table = pickle.load(open(q_table_file, "br"))

    # simulates the episodes
    frames = []
    for episode in tqdm.tqdm(range(episodes)):
        # resets environment
        observation, _ = env.reset()
        state_pos = np.digitize(observation[0], space_pos)
        state_vel = np.digitize(observation[1], space_vel)
        action = np.argmax(q_table[state_pos, state_vel, :])

        # single episode simulation
        while True:
            #Render to frames buffer
            frames.append(env.render())
            
            # runs single action step
            observation, reward, terminated, truncated, _ = env.step(action)
            state_pos = np.digitize(observation[0], space_pos)
            state_vel = np.digitize(observation[1], space_vel)
            action = np.argmax(q_table[state_pos, state_vel, :])

            # ends if done simulation
            if terminated or truncated:
                break

    # saves trained table
    save_frames_as_gif(frames, gif_file)


def save_frames_as_gif(frames, filename = None):
    # Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    animate = lambda i: patch.set_data(frames[i])
    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(filename, writer='imagemagick', fps=60)


# render running examples from trained table
run(episodes=5, q_table_file = "mcar_discrete_q_table.pkl", gif_file = "mcar_discrete_q_table.gif")