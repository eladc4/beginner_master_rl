
import random
import copy
import gym
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from utils import test_agent, plot_stats, seed_everything


# Create and prepare the environment
_env = gym.make('CartPole-v1', render_mode='rgb_array')
_env.reset()
# plt.imshow(_env.render())
# plt.show()

state_dims = _env.observation_space.shape[0]
num_actions = _env.action_space.n
print(f"CartPole env: State dimensions: {state_dims}, Number of actions: {num_actions}")

# select device
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


# Prepare the environment to work with PyTorch
class PreprocessEnv(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)

    # Wraps env.reset
    def reset(self):
        state, _ = self.env.reset()
        return torch.from_numpy(state).unsqueeze(dim=0).float().to(device), None

    # Wraps env.step
    def step(self, action, *args, **kwargs):
        next_state, reward, done, info1, info2 = self.env.step(int(action.item()))
        next_state = torch.from_numpy(next_state).unsqueeze(dim=0).float().to(device)
        reward = torch.Tensor([reward]).view(1, -1).float().to(device)
        done = torch.Tensor([done]).view(1, -1).float().to(device)
        return next_state, reward, done, info1, info2


env = PreprocessEnv(_env)

state, _ = env.reset()
action = torch.Tensor([0]).to(device)

next_state, reward, done, _, _ = env.step(action)
print(f"Sample state: {state}")
print(f"Next state: {next_state}, Reward: {reward}, Done: {done}")

# Create the Q-Network: $\hat q(s,a| \theta)$
q_network = nn.Sequential(
    nn.Linear(state_dims, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, num_actions)
)

# Create the target Q-Network: $\hat q(s, a|\theta_{targ})$
target_q_network = copy.deepcopy(q_network)
target_q_network = target_q_network.eval()

use_saved_model = False
if use_saved_model:
    q_network.load_state_dict(torch.load('sec9_q_network.pth'))
    target_q_network.load_state_dict(torch.load('sec9_target_q_network.pth'))

q_network.to(device)
target_q_network.to(device)


# Create the $\epsilon$-greedy policy: $\pi(s)$
def policy(state, epsilon=0.):
    if torch.rand(1) < epsilon:
        return torch.randint(num_actions, (1, 1)).to(device)
    else:
        av = q_network(state.to(device)).detach()
        return torch.argmax(av, dim=-1, keepdim=True)


# Create the Experience Replay buffer
# A simple buffer that stores transitions of arbitrary values, adapted from
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#training"
class ReplayMemory:

    def __init__(self, capacity=1000000):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.min_batches_to_sample = 10

    def insert(self, transition):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        assert self.can_sample(batch_size)

        batch = random.sample(self.memory, batch_size)
        batch = zip(*batch)

        aa=[]
        for items in batch:
            aa.append(torch.cat(items))

        return aa#[torch.cat(items) for items in batch]

    def can_sample(self, batch_size) -> bool:
        return len(self.memory) >= batch_size * self.min_batches_to_sample

    def __len__(self):
        return len(self.memory)


def deep_q_learning(q_network, policy, episodes,
                    alpha=0.0001, batch_size=32, gamma=0.99, epsilon=0.05):

    optim = AdamW(q_network.parameters(), lr=alpha)
    memory = ReplayMemory(capacity=1000000)
    stats = {'MSE Loss': [], 'Returns': []}

    for episode in tqdm(range(1, episodes + 1)):
        state, _ = env.reset()
        done = False
        ep_return = 0.
        loss_list = []

        while not done:
            action = policy(state, epsilon)
            next_state, reward, done, _, _ = env.step(action)
            memory.insert([state, action, reward, done, next_state])

            if memory.can_sample(batch_size):
                state_b, action_b, reward_b, done_b, next_state_b = memory.sample(batch_size)
                qsa_b = q_network(state_b).gather(1, action_b)

                next_qsa_b = target_q_network(next_state_b).max(dim=1, keepdim=True).values
                target_b = reward_b + (1 - done_b) * gamma * next_qsa_b

                loss = F.mse_loss(qsa_b, target_b.detach())
                q_network.zero_grad()
                loss.backward()
                optim.step()

                loss_list.append(loss.item())

            state = next_state
            ep_return += reward.item()

        stats['Returns'].append(ep_return)
        stats['MSE Loss'].append(sum(loss_list) / len(loss_list) if loss_list else 0)

        if episode % 10 == 0:
            target_q_network.load_state_dict(q_network.state_dict())

    return stats


if not use_saved_model:
    stats = deep_q_learning(q_network, policy, 500, batch_size=128)

    torch.save(q_network.state_dict(), 'sec9_q_network.pth')
    torch.save(target_q_network.state_dict(), 'sec9_target_q_network.pth')

    plot_stats(stats)

# test_agent(env, policy, episodes=2)