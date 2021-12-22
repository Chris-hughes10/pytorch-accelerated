from pytest import fixture, mark
from torch import nn

from pytorch_accelerated.finetuning import ModelFreezer, _module_is_batch_norm


class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(100, 100)
        self.block_1 = nn.Sequential(
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
        )
        self.block_2 = nn.Sequential(
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.Sequential(
                nn.Linear(100, 100),
                nn.BatchNorm1d(100),
                nn.ReLU(),
            ),
        )
        self.output_1 = nn.Linear(100, 10)
        self.output_2 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.input(x)
        x = self.block_1(x)
        x = self.block_2(x)
        out_1 = self.output_1(x)
        out_2 = self.output_2(x)
        return (out_1, out_2)


@fixture
def model():
    return TestModel()


def verify_layer_state(layer, frozen=True):
    params = list(layer.module.parameters(recurse=False))
    if params:
        requires_grad = {param.requires_grad for param in params}
        all_params_same_state = len(requires_grad) == 1
        all_params_correct_state = (not frozen) is list(requires_grad)[0]
        layer_state_correct = layer.is_frozen is frozen
    else:
        all_params_same_state = True
        all_params_correct_state = True
        if _module_is_batch_norm(layer.module):
            layer_state_correct = layer.is_frozen is frozen
        else:
            layer_state_correct = True

    return all_params_same_state and all_params_correct_state and layer_state_correct


@mark.parametrize(
    ["idxs", "expected_idxs"],
    [
        [(0, 4), (0, 4)],  # leaves idxs in correct format unchanged
        [(4, 0), (0, 4)],  # from_idx greater than to_idx
        [(-2, -1), (3, 4)],  # using neg idxs
        [(-1, -2), (3, 4)],  # reversed neg idxs
        [(0, -1), (0, 4)],  # using combination of pos, neg idxs
    ],
)
def test_can_convert_idxs(model, idxs, expected_idxs):
    freezer = ModelFreezer(model, freeze_batch_norms=True)

    actual_idxs = freezer._convert_idxs(*idxs)

    assert actual_idxs == expected_idxs


@mark.parametrize(
    ["from_index", "to_index"],
    [
        [0, 2],
        [0, 4],
        [1, 3],
    ],
)
def test_can_freeze_model(model, from_index, to_index):
    freezer = ModelFreezer(model, freeze_batch_norms=True)

    freezer.freeze(from_index=from_index, to_index=to_index)
    layers = freezer.get_layers()

    for layer in layers:
        if from_index <= layer.layer_group_idx[0] <= to_index:
            assert verify_layer_state(layer, frozen=True)
        else:
            assert verify_layer_state(layer, frozen=False)


@mark.parametrize(
    ["from_index", "to_index"],
    [
        [0, 2],
        [0, 4],
        [1, 3],
    ],
)
def test_can_freeze_model_except_batch_norm(model, from_index, to_index):
    freezer = ModelFreezer(model, freeze_batch_norms=False)

    freezer.freeze(from_index=from_index, to_index=to_index)
    layers = freezer.get_layers()

    for layer in layers:
        if from_index <= layer.layer_group_idx[0] <= to_index:
            if _module_is_batch_norm(layer.module):
                assert verify_layer_state(layer, frozen=False)
            else:
                assert verify_layer_state(layer, frozen=True)
        else:
            assert verify_layer_state(layer, frozen=False)


@mark.parametrize(
    ["from_index", "to_index"],
    [
        [0, 2],
        [0, 4],
        [1, 3],
    ],
)
def test_can_unfreeze_model(model, from_index, to_index):
    freezer = ModelFreezer(model, freeze_batch_norms=True)
    freezer.freeze(from_index=0, to_index=-1)

    freezer.unfreeze(from_index, to_index)
    layers = freezer.get_layers()

    for layer in layers:
        if from_index <= layer.layer_group_idx[0] <= to_index:
            assert verify_layer_state(layer, frozen=False)
        else:
            assert verify_layer_state(layer, frozen=True)


@mark.parametrize(
    ["from_index", "to_index"],
    [
        [0, 2],
        [0, 4],
        [1, 3],
    ],
)
def test_unfreeze_model_returns_correct_parameters_for_frozen_model(
    model, from_index, to_index
):
    freezer = ModelFreezer(model, freeze_batch_norms=True)
    freezer.freeze(from_index=0, to_index=-1)
    expected_params = {
        lg.layer_group_idx[0]: {"params": list(lg.module.parameters())}
        for lg in freezer.get_layer_groups()
        if from_index <= lg.layer_group_idx[0] <= to_index
    }

    actual_params = freezer.unfreeze(from_index, to_index)

    assert actual_params == expected_params


@mark.parametrize(
    ["from_index", "to_index"],
    [
        [0, 4],
    ],
)
def test_unfreeze_model_returns_correct_parameters_for_partially_frozen_model(
    model, from_index, to_index
):
    freezer = ModelFreezer(model, freeze_batch_norms=True)
    freezer.freeze(from_index=0, to_index=-2)
    expected_params = {
        lg.layer_group_idx[0]: {"params": list(lg.module.parameters())}
        for lg in freezer.get_layer_groups()
        if (from_index <= lg.layer_group_idx[0] <= to_index) and lg.is_frozen is True
    }

    actual_params = freezer.unfreeze(from_index, to_index)

    assert actual_params == expected_params
