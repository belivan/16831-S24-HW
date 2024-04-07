from typing import Union

import torch
from torch import nn

Activation = Union[str, nn.Module] # Union of string and nn.Module
# Union is a type that can be one of several types. It is used to define a type that can be one of several types.


_str_to_activation = {  # this is a dictionary that maps strings to activation functions
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
}


def build_mlp(
        input_size: int,
        output_size: int,
        n_layers: int,
        size: int,
        activation: Activation = 'tanh',
        output_activation: Activation = 'identity',
) -> nn.Module:
    """
        Builds a feedforward neural network

        arguments:
            n_layers: number of hidden layers
            size: dimension of each hidden layer
            activation: activation of each hidden layer

            input_size: size of the input layer
            output_size: size of the output layer
            output_activation: activation of the output layer

        returns:
            MLP (nn.Module)
    """
    if isinstance(activation, str):  # if the activation is a string, then we convert it to the corresponding activation function
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):  # if the output_activation is a string, then we convert it to the corresponding activation function
        output_activation = _str_to_activation[output_activation]

    # TODO: return a MLP. This should be an instance of nn.Module  [OK]
    # Note: nn.Sequential is an instance of nn.Module.
    # raise NotImplementedError
    
    layers = [nn.Linear(input_size, size), activation]  # layers is a list of layers, which is initialized with the first layer
    for _ in range(n_layers - 1):
        layers += [nn.Linear(size, size), activation]  # size: dimension of each hidden layer
    layers += [nn.Linear(size, output_size), output_activation]  # output_size: size of the output layer

    return nn.Sequential(*layers)
    # nn.Sequential is a container for layers, which is used to create a sequence of layers. * is used to unpack the list of layers


device = None


def init_gpu(use_gpu=True, gpu_id=0):
    global device
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda:" + str(gpu_id))
        print("Using GPU id {}".format(gpu_id))
    else:
        device = torch.device("cpu")
        print("GPU not detected. Defaulting to CPU.")


def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)


def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)


def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()
