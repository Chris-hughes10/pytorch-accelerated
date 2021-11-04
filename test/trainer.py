from pytest import fixture
from torch import optim, nn

from pytorch_accelerated.trainer import Trainer


class DummyTrainer(Trainer):
    def calculate_train_batch_loss(self, batch):
        pass

    def calculate_eval_batch_loss(self, batch):
        pass

    def _prepare_model_and_optimizer(self):
        pass


class SimpleModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)


@fixture
def model():
    return SimpleModel(10, 1)


@fixture
def optimizer(model):
    return optim.SGD(model.parameter(), lr=0.01)


def test_can_load_model_and_optimizer(model, optimizer):
    pass


def test_can_save_checkpoint():
    pass


def test_skip_eval_if_not_present():
    pass


def test_skip_scheduler_step_if_not_present():
    pass


def test_can_override_train_dataloader_kwargs():
    pass


def test_can_override_eval_dataloader_kwargs():
    pass


def test_model_is_in_correct_mode():
    pass


def test_gradient_accumulation():
    pass


def test_can_create_scheduler():
    pass

def test_can_inject_placeholders():
    pass
