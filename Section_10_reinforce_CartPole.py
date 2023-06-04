
## Import the necessary software libraries:

import os
import torch
import gym
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import nn as nn
from torch.optim import AdamW

from utils import test_policy_network, seed_everything, plot_stats, plot_action_probs
from parallel_env import ParallelEnv, ParallelWrapper


# select device
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


if __name__ == '__main__':
    ## Create and preprocess the environment

    ### Create the environment
    _env = gym.make('CartPole-v1', render_mode='rgb_array')

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


    env_fns = [lambda: create_env('CartPole-v1', rank) for rank in range(num_envs)]
    _penv = ParallelEnv(env_fns)

    _penv.reset()

    ### Prepare the environment to work with PyTorch

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

    state = penv.reset()
    _, reward, done, _ = penv.step(torch.zeros(num_envs, 1, dtype=torch.int32))
    print(f"State: {state}, Reward: {reward}, Done: {done}")

    ### Create the policy $\pi(s)$
    policy = nn.Sequential(
        nn.Linear(dims, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, actions),
        nn.Softmax(dim=-1)
    )
    policy = policy.to(device)


    ## Implement the algorithm
    def reinforce(policy, episodes, alpha=1e-4, gamma=0.99):

        optim = AdamW(policy.parameters(), lr=alpha)
        stats = {'Loss': [], 'Returns': []}

        for episode in tqdm(range(1, episodes+1)):

            state = penv.reset()
            done_b = torch.zeros((num_envs, 1)).to(device)
            transitions = []
            ep_return = torch.zeros((num_envs, 1)).to(device)

            while not (done_b == 1).all():
                action = policy(state).multinomial(1).detach()
                next_state, reward, done, _ = penv.step(action)
                transitions.append([state, action, (1-done_b)*reward])
                ep_return += reward
                done_b = torch.maximum(done_b, done)
                state = next_state

            G = torch.zeros((num_envs, 1)).to(device)
            for t, (state_t, action_t, reward_t) in reversed(list(enumerate(transitions))):
                G = reward_t + gamma * G
                probs_t = policy(state_t)
                log_probs_t = torch.log(probs_t + 1e-6)
                action_log_prob_t = log_probs_t.gather(1, action_t)

                entropy_t = -torch.sum(probs_t * log_probs_t, dim=-1, keepdim=True)
                gamma_t = gamma ** t

                pg_loss_t = - gamma_t * action_log_prob_t * G  # negative because we want gradient ascent
                total_loss_t = (pg_loss_t - 0.01 * entropy_t).mean()

                policy.zero_grad()
                total_loss_t.backward()
                optim.step()

            stats['Loss'].append(total_loss_t.item())
            stats['Returns'].append(ep_return.mean().item())

        return stats


    penv.reset()
    stats = reinforce(policy, 200)
    a=1










