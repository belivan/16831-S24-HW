import abc
import itertools
from typing import Any
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from rob831.infrastructure import pytorch_util as ptu
from rob831.policies.base_policy import BasePolicy
from torch.nn.modules.activation import MultiheadAttention


class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):
    # metaclass=abc.ABCMeta is a Python 3 feature that makes this an abstract class
    # meaning that it can't be instantiated directly, only subclasses can be
    # instantiated. This is a good way to ensure that you don't accidentally
    # create a policy that doesn't have the methods you need to implement.

    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.discrete = discrete
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        if self.discrete:
            self.logits_na = ptu.build_mlp(
                input_size=self.ob_dim,  # input size: observation dimension
                output_size=self.ac_dim,  # output size: action dimension
                n_layers=self.n_layers,  # number of layers, meaning number of hidden layers??
                size=self.size,
            )
            self.logits_na.to(ptu.device)  # move the model to the GPU
            self.mean_net = None  # mean_net is not used in the discrete case, which is the mean of the distribution
            self.logstd = None  # logstd is not used in the discrete case, which is log of the standard deviation
            self.optimizer = optim.Adam(self.logits_na.parameters(),
                                        self.learning_rate)
        else:
            self.logits_na = None
            self.mean_net = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.mean_net.to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
            )
            self.logstd.to(ptu.device)
            self.optimizer = optim.Adam(
                itertools.chain([self.logstd], self.mean_net.parameters()),
                self.learning_rate
            )  # chain is used to combine the parameters of the mean_net and logstd
            # into a single iterable so that they can be passed to the optimizer
            # as a single argument
            # more info: https://docs.python.org/3/library/itertools.html#itertools.chain

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    def get_action(self, obs: np.ndarray) -> np.ndarray:  # -> np.ndarray means that the function returns a numpy array
        # obs: (ob_dim, ) or (batch_size, ob_dim)
        # convert the observation to a tensor and move it to the GPU
        # if it's not already there

        if len(obs.shape) > 1:
            observation = obs  # if the observation is a batch of observations, then we don't need to add a dimension
        else:
            observation = obs[None]  # if the observation is a single observation, then we need to add a dimension

        # TODO return the action that the policy prescribes
        observation = ptu.from_numpy(observation)
        #action = self.forward(observation)

        # if self.discrete:
        #     # For discrete action spaces, action_distribution is a Categorical
        #     # probs = F.softmax(action, dim=-1)
        #     logits = self.logits_na(observation)
        #     distribution = distributions.Categorical(logits=logits)  # probs = probs)
        #     sampled_action = distribution.sample()
        # else:
        #     # For continuous action spaces, action_distribution is a Normal
        #     std = torch.exp(self.logstd)
        #     distribution = distributions.Normal(action, std)
        #     sampled_action = distribution.rsample()  # Reparameterized sample for differentiability
            # for more info on rsample: https://pytorch.org/docs/stable/distributions.html#torch.distributions.Distribution.rsample
        mean = self.forward(observation)

        std = self.logstd.exp().expand_as(mean)
        action = distributions.Normal(mean, std).rsample()

        return ptu.to_numpy(action)

        # raise NotImplementedError

    # update/train this policy
    def update(self, observations, actions, **kwargs):
        # observations: (batch_size, ob_dim)
        # actions: (batch_size, ac_dim)
        # **kwargs is a dictionary that contains any extra arguments that are passed to the function
        # to update the policy, we can do the following:
        # 1. calculate the gradient of the loss with respect to the parameters of the policy
        # 2. update the parameters of the policy using the gradients
        # 3. return the loss

        # TODO: update the policy and return the loss (I wrote this comment)
        # Helpful functions:
        # self.optimizer.zero_grad() # clear previous gradients
        # loss.backward() # compute gradients
        # self.optimizer.step() # update the parameters of the policy

        # Example of implementation:

        # loss = F.mse_loss(self(observations), actions)
        # F is the functional module of PyTorch, which contains functions that are used to define the loss

        # If self(observations) return a distribution then we can use the following code
        observations = ptu.from_numpy(observations)
        actions = ptu.from_numpy(actions)

        predicted_actions = self.forward(observations)

        if self.discrete:
            probs = F.softmax(predicted_actions, dim=-1)
            distribution = distributions.Categorical(probs=probs)# logits=predicted_actions)
        else:
            std = torch.exp(self.logstd)
            distribution = distributions.Normal(predicted_actions, std)
        
        loss = -distribution.log_prob(actions).mean()
        # log_prob is used to calculate the log probability of the actions
        # on log probabilities: https://en.wikipedia.org/wiki/Log_probability
        # mean is used to calculate the mean of the log probabilities
        # - is used to make the loss negative, because we want to maximize the log probabilities

        self.optimizer.zero_grad()  # clear previous gradients
        loss.backward()  # compute gradients
        self.optimizer.step()  # update the parameters of the policy

        return {
            # You can add extra logging information here, but keep this line
            'Training Loss': ptu.to_numpy(loss),
        }
        # raise NotImplementedError

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor) -> Any:
        # TODO: return the action distribution (I wrote this comment)
        # Convert ovservation to Torch Tensor:
        # raise NotImplementedError
        # observation: (batch_size, ob_dim)
        # return the action distribution
        #observation = ptu.from_numpy(observation.astype(np.float32))
        #observation = observation.to(ptu.device)  # move the observation to the GPU, is it necessary?
        if self.discrete:
            return self.logits_na(observation)  # return the logits, which are the unnormalized log probabilities
            # here is description of log probabilities: https://en.wikipedia.org/wiki/Log_probability
            # mean = F.softmax(action, dim=-1)  # dim=-1 means that we are applying softmax along the last dimension
            # softmax is used to convert the logits to probabilities
            # std = None  # std is not used in the discrete case
            # gaussian = distributions.Categorical(mean)
            # return gaussian

        return self.mean_net(observation)  # return the mean of the action distribution
        # mean is the expected value of the action distribution
        # useful for continuous action spaces

        # mean = self.mean_net(observation)
        # std = torch.exp(self.logstd)  # logstd is the log of the standard deviation,
        # # torch.exp is used to get the standard deviation
        # gaussian = distributions.Normal(mean, std)
        # return gaussian


#####################################################
#####################################################

class MLPPolicySL(MLPPolicy):  # SL stands for supervised learning
    def __init__(self, ac_dim, ob_dim, n_layers, size, **kwargs):
        super().__init__(ac_dim, ob_dim, n_layers, size, **kwargs)
        self.loss = nn.MSELoss()

    def update(
            self, observations, actions,
            adv_n=None, acs_labels_na=None, qvals=None
    ):
        # TODO: update the policy and return the loss
        observations = ptu.from_numpy(observations)
        actions = ptu.from_numpy(actions)

        mean = self.forward(observations)

        std = self.logstd.exp().expand_as(mean)
        predicted_actions = distributions.Normal(mean, std).rsample()

        loss = self.loss(predicted_actions, actions)
        # we forward the observations through the policy to get the predicted actions
        # then we calculate the loss between the predicted actions and the true actions
        self.optimizer.zero_grad()  # clear previous gradients
        loss.backward()
        self.optimizer.step()

        return {
            # You can add extra logging information here, but keep this line
            'Training Loss': ptu.to_numpy(loss),
        }
