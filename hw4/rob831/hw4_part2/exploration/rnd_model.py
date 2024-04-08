from rob831.hw4_part2.infrastructure import pytorch_util as ptu
from .base_exploration_model import BaseExplorationModel
import torch.optim as optim
from torch import nn
import torch

def init_method_1(model):
    model.weight.data.uniform_()
    model.bias.data.uniform_()

def init_method_2(model):
    model.weight.data.normal_()
    model.bias.data.normal_()


class RNDModel(nn.Module, BaseExplorationModel):
    def __init__(self, hparams, optimizer_spec, **kwargs):
        super().__init__(**kwargs)
        self.ob_dim = hparams['ob_dim']
        self.output_size = hparams['rnd_output_size']
        self.n_layers = hparams['rnd_n_layers']
        self.size = hparams['rnd_size']
        self.optimizer_spec = optimizer_spec

        # <TODO>: Create two neural networks:
        # 1) f, the random function we are trying to learn
        # 2) f_hat, the function we are using to learn f
        self.f = ptu.build_mlp(
            input_size=self.ob_dim,
            output_size=self.output_size,
            n_layers=self.n_layers,
            size=self.size,
            init_method=init_method_1
        ).to(ptu.device)

        self.f_hat = ptu.build_mlp(
            input_size=self.ob_dim,
            output_size=self.output_size,
            n_layers=self.n_layers,
            size=self.size,
            init_method=init_method_2
        ).to(ptu.device)

        # OptimizerSpec(
        # constructor=optim.Adam,
        # optim_kwargs=dict(
        #     lr=1e-3,
        #     eps=1e-4
        # ),
        # learning_rate_schedule=lambda t: lr_schedule.value(t),
        # )
        self.optimizer = self.optimizer_spec.constructor(
            self.f_hat.parameters(),
            **optimizer_spec.optim_kwargs
        )
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            self.optimizer_spec.learning_rate_schedule
        )

    def forward(self, ob_no):
        # <TODO>: Get the prediction error for ob_no
        # HINT: Remember to detach the output of self.f!
        error = (self.f(ob_no).detach() - self.f_hat(ob_no)).norm(dim=1)
        return error

    def forward_np(self, ob_no):
        ob_no = ptu.from_numpy(ob_no)
        error = self.forward(ob_no)
        return ptu.to_numpy(error)

    def update(self, ob_no):
        # <TODO>: Update f_hat using ob_no
        # Hint: Take the mean prediction error across the batch
        if not isinstance(ob_no, torch.Tensor):
            ob_no = ptu.from_numpy(ob_no)

        error = self.forward(ob_no)
        mean_error = error.mean()  # / ob_no.shape[0]

        self.optimizer.zero_grad()
        mean_error.backward()
        self.optimizer.step()

        self.scheduler.step()

        return ptu.to_numpy(mean_error)
