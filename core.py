import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):
    
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.



class MLPActorCritic(nn.Module):


    def __init__(self, observation_space, action_space, 
                 hidden_sizes=(64,64), activation=nn.Tanh):
        super().__init__()

        #obs_dim = observation_space.shape[0]
        obs_dim = np.prod(observation_space['image'].shape)

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        # build value function
        self.v  = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]

class RNDNetwork(nn.Module):

    def __init__(self, obs_dim, hidden_dim, feature_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
    
    def forward(self, obs):
        return self.model(obs)

class RND:

    def __init__(self, obs_dim, hidden_dim, feature_dim):
        self.target_net = RNDNetwork(obs_dim, hidden_dim, feature_dim)
        self.pred_net = RNDNetwork(obs_dim, hidden_dim, feature_dim)

    def compute_loss(self, obs):
        #print(f'obs shape = {obs.shape}')
        with torch.no_grad():
            feature_t = self.target_net(obs)
        feature_p = self.pred_net(obs)
        return nn.MSELoss()(feature_t, feature_p)

    def get_param(self):
        return self.pred_net.parameters()


## Intrinsic Curiosity Module Implementation

class StateEmbeddingNet(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.obs_dim = obs_dim

        self.feat_extract = nn.Sequential( 
            nn.Conv2d(in_channels=self.obs_dim[2], out_channels=32, kernel_size=(3, 3), stride=2, padding=1),
            nn.elu(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1),
            nn.elu(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1),
            nn.elu(),
        )

    def forward(self, obs):
        T, B, *_ = obs.shape

        x = torch.flatten(obs, 0, 1)
        x = x.float() / 255.0
        x = x.transpose(1, 3)
        x = self.feat_extract(x)

        obs_embedding = x.view(T, B, -1)
        return obs_embedding


class ForwardDynamicsNet(nn.Module):
    def __init__(self, act_dim):
        super().__init__()

        self.act_dim = act_dim

        self.forward_dynamics = nn.Sequential(
            nn.Linear(128 + self.act_dim, 256),
            nn.ReLU(),
        )

        self.fd_out = nn.Linear(256, 128)

    def forward(self, state_embedding, action):
        action_one_hot = F.one_hot(action, num_classes=self.act_dim).float()
        inputs = torch.cat((state_embedding, action_one_hot), dim=2)
        next_state_emb = self.fd_out(self.forward_dynamics(inputs))
        return next_state_emb

    def compute_loss(self, next_state_embedding_pred, next_state_embedding):
        loss = torch.norm(next_state_embedding_pred - next_state_embedding, dim=2, p=2)
        return torch.sum(torch.mean(loss, dim=1))

class InverseDynamicsNet(nn.Module):
    def __init__(self, act_dim):
        super().__init__()

        self.act_dim = act_dim

        self.inverse_dynamics = nn.Sequential(
            nn.Linear(2 * 128, 256),
            nn.ReLU(),
        )

        self.id_out = nn.Linear(256, self.act_dim)

    def forward(self, state_embedding, next_state_embedding):
        inputs = torch.cat((state_embedding, next_state_embedding), dim=2)
        action_logits = self.id_out(self.inverse_dynamics(inputs))
        return action_logits

    def compute_loss(self, action_pred, action):
        loss = F.nll_loss(
            F.log_softmax(torch.flatten(action_pred, 0, 1), dim=-1), 
            target=torch.flatten(action, 0, 1), 
            reduction='none'
        )
        loss = loss.view_as(action)
        return torch.sum(torch.mean(loss, dim=1))


class ICM:
    def __init__(self, obs_dim,  act_dim, beta=0.2):
        self.beta = beta
        self.state_embedding = StateEmbeddingNet(obs_dim)
        self.forward_model = ForwardDynamicsNet(act_dim)
        self.inverse_model = InverseDynamicsNet(act_dim)
    
    def calculate_intrinsic(self, state_embedding, action, next_state_embedding):
        with torch.no_grad():
            next_state_embedding_pred = self.forward_model(state_embedding, action)
            intrinsic = torch.norm(next_state_embedding_pred - next_state_embedding, dim=2, p=2)
        return intrinsic

    def compute_loss(self, state, next_state, action):
        state_embedding = self.state_embedding(state)
        next_state_embedding = self.state_embedding(next_state)
        
        next_state_embedding_pred = self.forward_model(state_embedding, action)
        action_pred = self.inverse_model(state_embedding, next_state_embedding)

        loss_forward = self.forward_model.compute_loss(next_state_embedding_pred, next_state_embedding)
        loss_inverse = self.inverse_model.compute_loss(action_pred, action)

        return (1 - self.beta)*loss_inverse + self.beta*loss_forward