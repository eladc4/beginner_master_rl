
import os
import torch
import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import nn as nn
from torch.optim import AdamW
import torch.nn.functional as F

from utils import test_policy_network, seed_everything, plot_stats
from parallel_env import ParallelEnv, ParallelWrapper


if __name__ == '__main__':

    # select device
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'


    #Create and preprocess the environment

    #Create the environment
    _env = gym.make('Acrobot-v1', render_mode='rgb_array')

    dims = _env.observation_space.shape[0]
    actions = _env.action_space.n

    print(f"State dimensions: {dims}. Actions: {actions}")
    print(f"Sample state: {_env.reset()}")

    # plt.imshow(_env.render())

    ### Parallelize the environment
    num_envs = os.cpu_count()


    def create_env(env_name, seed):
        env = gym.make(env_name, render_mode='rgb_array')
        seed_everything(env, seed=seed)
        return env


    env_fns = [lambda: create_env('Acrobot-v1', rank) for rank in range(num_envs)]
    _penv = ParallelEnv(env_fns)

    _penv.reset()

    # Prepare the environment to work with PyTorch
    class PreprocessEnv(ParallelWrapper):

        def __init__(self, _penv):
            ParallelWrapper.__init__(self, _penv)

        # Wraps penv.reset
        def reset(self):
            state = self.venv.reset()
            return torch.from_numpy(state).float().to(device)

        # Wraps penv.step_async
        def step_async(self, actions):
            actions = actions.squeeze().cpu().numpy()
            self.venv.step_async(actions)

        def step_wait(self):
            next_state, reward, done, info1 = self.venv.step_wait()
            next_state = torch.from_numpy(next_state).float().to(device)
            reward = torch.Tensor([reward]).view(-1, 1).float().to(device)
            done = torch.Tensor([done]).view(-1, 1).float().to(device)
            return next_state, reward, done, info1


    penv = PreprocessEnv(_penv)

    # Create the policy $\pi(s)$
    policy = nn.Sequential(
        nn.Linear(dims, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, actions),
        nn.Softmax(dim=-1)
    )
    policy = policy.to(device)

    # Create the value network v(s)
    value_net = nn.Sequential(
        nn.Linear(dims, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1))
    value_net = value_net.to(device)

    # Implement the algorithm
    def actor_critic(policy, value_net, episodes, alpha=1e-4, gamma=0.99):

        policy_optim = AdamW(policy.parameters(), lr=alpha)
        value_optim = AdamW(value_net.parameters(), lr=alpha)
        stats = {'Actor Loss': [], 'Critic Loss': [], 'Returns': []}

        for episode in tqdm(range(1, episodes+1)):

            state = penv.reset()
            done_b = torch.zeros((num_envs, 1)).to(device)
            ep_return = torch.zeros((num_envs, 1)).to(device)
            I = 1.0

            while not (done_b == 1).all():
                action = policy(state).multinomial(1).detach()
                next_state, reward, done, _ = penv.step(action)

                value = value_net(state)
                target = reward + (1 - done) * gamma * value_net(next_state).detach()
                critic_loss = F.mse_loss(value, target)
                value_net.zero_grad()
                critic_loss.backward()
                value_optim.step()

                advantage = (target - value).detach()
                probs = policy(state)
                log_probs = torch.log(probs + 1e-6)
                action_log_prob = log_probs.gather(1, action)
                entropy = - torch.sum(probs * log_probs, dim=-1, keepdim=True)
                actor_loss = - I * action_log_prob * advantage - 0.01 * entropy
                actor_loss = actor_loss.mean()
                policy.zero_grad()
                actor_loss.backward()
                policy_optim.step()

                ep_return += reward
                done_b = torch.maximum(done_b, done)
                state = next_state
                I = I * gamma

            stats['Actor Loss'].append(actor_loss.item())
            stats['Critic Loss'].append(critic_loss.item())
            stats['Returns'].append(ep_return.mean().item())

        return stats


    stats = actor_critic(policy, value_net, 200)
