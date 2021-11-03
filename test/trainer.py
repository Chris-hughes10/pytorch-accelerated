from pytest import fixture
from torch import optim
from torchvision import models


@fixture
def model():
    return models.resnet18(pretrained=False)

@fixture
def optimizer(model):
    return optim.SGD(model.parameter(), lr=0.01)

def test_can_load_model_and_optimizer(model, optimizer):
    pass

def test_skip_eval_if_not_present():
    pass

def test_skip_scheduler_step_if_not_present():
    pass

def test_can_override_dataloader_kwargs():
    pass