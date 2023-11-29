import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import pickle

import torch
import torch.nn as nn


HID_SIZE = 128
class MCARContinuous_NormalReinforce(nn.Module):
    def __init__(self):
        super(MCARContinuous_NormalReinforce, self).__init__()
        self.base = nn.Sequential(
            nn.Linear(2, HID_SIZE),
            nn.ReLU())
        self.m = nn.Sequential(
            nn.Linear(HID_SIZE, 1),
            nn.Tanh())
        self.s = nn.Sequential(
            nn.Linear(HID_SIZE, 1),
            nn.Softplus())
    
    def forward(self, x):
        base_out = self.base(x)
        return self.m(base_out), self.s(base_out)
    
def param_to_action(param):
    m = param[0].data.cpu().numpy()
    s = torch.sqrt(param[1]).data.cpu().numpy()
    return np.clip(np.random.normal(m, s), -1, 1)
    
def calc_qvals(rewards, gamma):
    res = []
    sum_r = 0
    for r in reversed(rewards):
        sum_r *= gamma
        sum_r += r
        res.append(sum_r)
    return list(reversed(res))
    
def calc_logprob(m, s, actions_v):
    p1 = -((m - actions_v) ** 2) / (2*s.clamp(min=1e-3))
    p2 = -torch.log(torch.sqrt(2 * np.pi * s))
    return p1 + p2
    

def run(epochs = 100, episodes = 10, train = True, render = True, lr = 0.0001, gamma = 0.9,
        model_file = None, rewards_plot_file = None, actions_hist_file = None):
    # initializes the environment
    env = gym.make('MountainCarContinuous-v0', render_mode = "human" if render else None)

    # initializes the network
    if train:
        mcar_net = MCARContinuous_NormalReinforce()
        # optimizer = torch.optim.SGD(mcar_net.parameters(), lr=lr)
        optimizer = torch.optim.Adam(mcar_net.parameters(), lr=lr)
    elif model_file:
        mcar_net = MCARContinuous_NormalReinforce()
        mcar_net.load_state_dict(torch.load(model_file))
    else:
        raise Exception("No trained model file was given!")

    # saves total rewards
    rewards = np.zeros((epochs, episodes))
    actions = []

    # monte-carlo over epochs
    for epoch in tqdm.tqdm(range(epochs)):
        # simulates N episodes
        # epoch_losses = 0
        for episode in range(episodes):
            # accumulates transitions
            transitions = []

            # resets environment
            observation, _ = env.reset()

            # single episode simulation
            while True:
                # runs single action step
                action = param_to_action(mcar_net(torch.from_numpy(observation)))
                observation, reward, terminated, _, _ = env.step(action)
                
                # updates sistem states
                reward = -1
                actions.append(action[0])
                rewards[epoch, episode] += reward
                transitions.append([observation[0], observation[1], action[0], reward])
                
                # ends if done simulation
                if terminated or len(transitions) > 10000:
                    break
        
            transitions = torch.from_numpy(np.array(transitions))
            qvals = calc_qvals(transitions[:, 3], gamma)
            qvals = torch.from_numpy(np.array(qvals))
            m, s = mcar_net(transitions[:, 0:2].float())
            episode_loss = -qvals*calc_logprob(m, s, transitions[:, 2])
            episode_loss = episode_loss.mean()
            # epoch_losses += episode_loss
            
        
        # update if training
        if train:
            episode_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    # saves trained table
    if model_file:
        torch.save(mcar_net.state_dict(), model_file)

    # saves rewards plot
    if rewards_plot_file:
        rewards = rewards.flatten()
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


if __name__ == "__main__":
    # trains model and stores q-table and rewards
    run(epochs=1000, episodes=1, train=True, render=False,
        model_file = "mcar_continuous_reinforce_model",
        rewards_plot_file = "mcar_continuous_learn_reinforce_rewards_plot")
    
    # generate actions histogram from trained table
    run(epochs=200, episodes=1, train=False, render=False,
        model_file = "mcar_continuous_reinforce_model",
        rewards_plot_file = "mcar_continuous_trained_reinforce_rewards_plot",
        actions_hist_file = "mcar_continuous_trained_reinforce_actions_hist")

    # render running examples from trained table
    run(epochs=5, episodes=1, train=False, render=True,
        model_file = "mcar_continuous_reinforce_model")
        

