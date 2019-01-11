import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)       # replay buffer size
BATCH_SIZE = 128             # minibatch size
GAMMA = 0.99                 # discount factor
TAU = 1e-3                   # for soft update of target parameters
LR_ACTOR = 1e-4              # learning rate of the actor
LR_CRITIC = 1e-4             # learning rate of the critic
WEIGHT_DECAY = 0             # L2 weight decay
SIGMA = 0.1                  # OU Noise sigma
ACTOR_NN_SIZE = [400, 300]   # dimension of hidden layers for actor fully connected NN
CRITIC_NN_SIZE = [400, 300]  # dimension of hidden layers for critic fully connected NN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DDPGAgent:
    """DDPG agent implementation."""
    
    def __init__(self, state_size, action_size, random_seed,
                 buffer_size=BUFFER_SIZE,
                 batch_size=BATCH_SIZE,
                 gamma=GAMMA,
                 tau=TAU,
                 lr_actor=LR_ACTOR,
                 lr_critic=LR_CRITIC,
                 weight_decay=WEIGHT_DECAY,
                 sigma=SIGMA,
                 actor_nn_size=ACTOR_NN_SIZE,
                 critic_nn_size=CRITIC_NN_SIZE,
                 batch_norm=True,
                 clip_grad_norm=True):
        """
        Initialization of the Agent
        :param state_size         (int):   dimension of each state
        :param action_size        (int):   dimension of each action
        :param random_seed        (int):   random seed
        :param buffer_size        (int):   number of samples that the replay buffer can store
        :param batch_size         (int):   number of samples used for learning for each learning step
        :param gamma            (float):   reward discount factor of the MDP problem
        :param tau              (float):   soft update factor, between 0 and 1, varies how fast the target network are updated
        :param lr_actor         (float):   learning rate for the actor
        :param lr_critic        (float):   learning rate for the critic
        :param weight_decay     (float):   weight decay regularization factor
        :param sigma            (float):   OU noise process randomness weight
        :param actor_nn_size  [int,int]:   2 dim array defining the number of units in the actor NN for the two fc layers
        :param critic_nn_size [int,int]:   2 dim array defining the number of units in the critic NN for the two fc layers
        :param batch_norm        (bool):   flag to control the use of batch normalization
        :param clip_grad_norm    (bool):   flag to control the use of critic backprop updated gradient clipping
        """
        # Hyperparameters
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.weight_decay = weight_decay
        self.sigma = sigma
        self.actor_nn_size = actor_nn_size
        self.critic_nn_size = critic_nn_size
        self.batch_norm = batch_norm
        self.clip_grad_norm = clip_grad_norm

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed, self.actor_nn_size[0], self.actor_nn_size[1], self.batch_norm).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed, self.actor_nn_size[0], self.actor_nn_size[1], self.batch_norm).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.lr_actor)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed, self.critic_nn_size[0], self.critic_nn_size[1], self.batch_norm).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed, self.critic_nn_size[0], self.critic_nn_size[1], self.batch_norm).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.lr_critic,
                                           weight_decay=self.weight_decay)

        # Noise process
        self.noise = OUNoise(action_size, random_seed, sigma=self.sigma)

        # Replay memory
        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, random_seed)
    
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences, self.gamma)

    def act(self, state, add_noise=True, noise_damping=1):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample() * noise_damping

        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)

    @staticmethod
    def soft_update(local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def load_weights(self, actor_weights_file, critic_weights_file):
        self.actor_local.load_state_dict(torch.load(actor_weights_file))
        self.actor_target.load_state_dict(torch.load(actor_weights_file))

        self.critic_local.load_state_dict(torch.load(critic_weights_file))
        self.critic_target.load_state_dict(torch.load(critic_weights_file))

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
