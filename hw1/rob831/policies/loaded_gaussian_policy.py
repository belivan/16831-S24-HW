import numpy as np

from rob831.infrastructure import pytorch_util as ptu
from .base_policy import BasePolicy
from torch import nn
import torch
import pickle

# This is a policy class that loads in a particular type of policy and queries it.
# Lets break down the code:
# 1. The class inherits from BasePolicy and nn.Module. This is because it is a policy and it is a neural network.
# 2. The __init__ method loads in a policy from a file. The policy is a Gaussian policy, which is a policy that outputs
#    a mean and a standard deviation. The policy is loaded from a file using pickle.
# 3. The forward method takes in an observation and returns the mean and standard deviation of the action distribution.
# 4. The get_action method takes in an observation and returns a sample from the action distribution.
# 5. The save method saves the policy to a file.
# 6. The update method raises a NotImplementedError. This is because this policy class is only for querying a policy,
#    not for training it.


def create_linear_layer(W, b) -> nn.Linear:
    # in this function, we create a linear layer with the given weights and biases
    out_features, in_features = W.shape
    linear_layer = nn.Linear(
        in_features,
        out_features,
    )
    linear_layer.weight.data = ptu.from_numpy(W.T)
    linear_layer.bias.data = ptu.from_numpy(b[0])
    return linear_layer


def read_layer(l):
    # In this layer, we read the weights and biases from a layer
    # The layer is a dictionary with keys 'W' and 'b'
    # The weights and biases are stored as numpy arrays
    # We convert the weights and biases to torch tensors and return them
    assert list(l.keys()) == ['AffineLayer']  # the layer should be an AffineLayer
    assert sorted(l['AffineLayer'].keys()) == ['W', 'b']
    return l['AffineLayer']['W'].astype(np.float32), l['AffineLayer'][
        'b'].astype(np.float32)


class LoadedGaussianPolicy(BasePolicy, nn.Module):
    def __init__(self, filename, **kwargs):
        super().__init__(**kwargs)

        with open(filename, 'rb') as f:
            data = pickle.loads(f.read())
            # f is a file object. We read the file and load the data using pickle.
            # the file contains a dictionary with keys 'nonlin_type' and 'GaussianPolicy'

        self.nonlin_type = data['nonlin_type']
        if self.nonlin_type == 'lrelu':
            self.non_lin = nn.LeakyReLU(0.01)
        elif self.nonlin_type == 'tanh':
            self.non_lin = nn.Tanh()
        else:
            raise NotImplementedError()
        policy_type = [k for k in data.keys() if k != 'nonlin_type'][0]

        assert policy_type == 'GaussianPolicy', (
            'Policy type {} not supported'.format(policy_type)
        )  # the policy type should be GaussianPolicy
        self.policy_params = data[policy_type]

        assert set(self.policy_params.keys()) == {
            'logstdevs_1_Da', 'hidden', 'obsnorm', 'out'
        }  # the policy_params should have these keys

        # Build the policy. First, observation normalization.
        # These are the mean and standard deviation of the observations
        assert list(self.policy_params['obsnorm'].keys()) == ['Standardizer']
        obsnorm_mean = self.policy_params['obsnorm']['Standardizer']['mean_1_D']
        obsnorm_meansq = self.policy_params['obsnorm']['Standardizer'][
            'meansq_1_D']
        obsnorm_stdev = np.sqrt(
            np.maximum(0, obsnorm_meansq - np.square(obsnorm_mean)))
        print('obs', obsnorm_mean.shape, obsnorm_stdev.shape)

        self.obs_norm_mean = nn.Parameter(ptu.from_numpy(obsnorm_mean))
        # nn.Parameter is a tensor that is a parameter of the model
        self.obs_norm_std = nn.Parameter(ptu.from_numpy(obsnorm_stdev))
        self.hidden_layers = nn.ModuleList()

        # Hidden layers next
        assert list(self.policy_params['hidden'].keys()) == ['FeedforwardNet']
        # the hidden layers should be a FeedforwardNet, which is a type of neural network, with one layer
        layer_params = self.policy_params['hidden']['FeedforwardNet']
        # layer_params is a dictionary with keys as the layer names and values as the layer parameters
        # values are weights and biases
        # layer names can be 'l1', 'l2', etc.
        # We sort the layer names and iterate through them
        for layer_name in sorted(layer_params.keys()):
            l = layer_params[layer_name]
            W, b = read_layer(l)
            linear_layer = create_linear_layer(W, b)
            self.hidden_layers.append(linear_layer)
            # we create a linear layer and add it to the list of hidden layers

        # Output layer
        W, b = read_layer(self.policy_params['out'])
        self.output_layer = create_linear_layer(W, b)
        # first we read the weights and biases, then create a linear layer with those weights and biases

    def forward(self, obs):
        # This method takes in an observation and returns the mean and standard deviation of the action distribution
        normed_obs = (obs - self.obs_norm_mean) / (self.obs_norm_std + 1e-6)
        h = normed_obs  # input to the first layer is the normalized observation
        # h is the output of the previous layer, that is, the input to the next layer
        for layer in self.hidden_layers:
            h = layer(h)
            h = self.non_lin(h)
        return self.output_layer(h)

    ##################################

    def update(self, obs_no, acs_na, adv_n=None, acs_labels_na=None):
        raise NotImplementedError("""
            This policy class simply loads in a particular type of policy and
            queries it. Do not try to train it.
        """)

    def get_action(self, obs):
        if len(obs.shape) > 1:  # if the observation is a 2D array
            observation = obs  # we use the observation as is
        else:
            observation = obs[None, :]  # if the observation is a 1D array, we add a dimension to it
        observation = ptu.from_numpy(observation.astype(np.float32))
        # float32 is the data type, that is needed for the neural network
        action = self(observation)
        # we pass the observation to the policy and get the mean and standard deviation of the action distribution
        # self(observation) is equivalent to self.forward(observation)
        # why: https://stackoverflow.com/questions/5824881/python-call-a-function-from-string-name
        return ptu.to_numpy(action)

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)
