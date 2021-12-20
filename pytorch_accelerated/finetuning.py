from collections import namedtuple, defaultdict
from pprint import pprint
from typing import Optional, Generator

import torch
from torch import nn
from torch.optim import Optimizer

BN_TYPES = (
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm3d,
    torch.nn.LazyBatchNorm1d,
    torch.nn.LazyBatchNorm2d,
    torch.nn.LazyBatchNorm3d,
    torch.nn.SyncBatchNorm,
)


def filter_params(module: torch.nn.Module, train_bn: bool = True) -> Generator:
    """Yields the trainable parameters of a given module.
    Args:
        module: A given module
        train_bn: If True, leave the BatchNorm layers in training mode
    Returns:
        Generator
    """
    children = list(module.children())
    if not children:
        if not (isinstance(module, BN_TYPES) and train_bn):
            for param in module.parameters():
                if param.requires_grad:
                    yield param
    else:
        for child in children:
            for param in filter_params(module=child, train_bn=train_bn):
                yield param


def _unfreeze_and_add_param_group(
    module: torch.nn.Module,
    optimizer: Optimizer,
    lr: Optional[float] = None,
    train_bn: bool = True,
):
    """Unfreezes a module and adds its parameters to an optimizer."""
    # _make_trainable(module)
    params_lr = optimizer.param_groups[0]["lr"] if lr is None else float(lr)
    optimizer.add_param_group(
        {
            "params": filter_params(module=module, train_bn=train_bn),
            "lr": params_lr / 10.0,
        }
    )


def get_trainable_parameters(
    module: torch.nn.Module, train_bn: bool = True
) -> Generator:
    """Yields the trainable parameters of a given module.
    Args:
        module: A given module
        train_bn: If True, leave the BatchNorm layers in training mode
    Returns:
        Generator
    """
    children = list(module.children())
    if not children:
        # is leaf
        if not (isinstance(module, BN_TYPES) and train_bn):
            for param in module.parameters():
                if param.requires_grad:
                    yield param
    else:
        for child in children:
            for param in get_trainable_parameters(module=child, train_bn=train_bn):
                yield param


class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
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


Layer = namedtuple("Layer", ["layer_group_idx", "module", "is_frozen"])
LayerGroup = namedtuple("LayerGroup", ["layer_group_idx", "module", "is_frozen"])


class ModelFreezer:
    def __init__(self, model, freeze_batch_norms=False):
        self.model = model
        set_requires_grad(model.parameters(), value=True)
        self._layer_groups, self._layers = get_layer_groups(model)
        self.freeze_bn = freeze_batch_norms
        self.num_groups = len(self._layer_groups)

    def get_layer_groups(self):
        layer_groups = []

        for group_idx, layer_group_module in self._layer_groups.items():
            params = {
                not param.requires_grad for param in layer_group_module.parameters()
            }
            frozen = True if True in params else False

            layer_groups.append(
                LayerGroup(
                    (group_idx, group_idx - self.num_groups), layer_group_module, frozen
                )
            )

        return layer_groups

    def get_layers(self):
        layers = []

        for group_idx, layer in self._layers:
            frozen_status = {
                param.requires_grad for param in layer.parameters(recurse=False)
            }
            if len(frozen_status) > 1:
                raise ValueError
            elif len(frozen_status) == 1:
                layers.append(
                    Layer(
                        (group_idx, group_idx - self.num_groups),
                        layer,
                        not list(frozen_status)[0],
                    )
                )
            else:
                # layer has no parameters
                layers.append(
                    Layer(
                        (group_idx, group_idx - self.num_groups),
                        layer,
                        not layer.training,
                    )
                )

        return layers

    def freeze(self, from_index=0, to_index=-2, set_eval=True):
        self.__freeze_unfreeze(
            from_index, to_index, freeze=True, toggle_train_eval=set_eval
        )

    def unfreeze(self, from_index=-1, to_index=0):
        unfrozen_params = self.__freeze_unfreeze(from_index, to_index, freeze=False)
        return unfrozen_params

    def __freeze_unfreeze(
        self,
        from_layer_group_index,
        to_layer_group_index,
        freeze=True,
        toggle_train_eval=True,
    ):
        modified_parameters = defaultdict(list)
        set_grad_value = not freeze
        layers = self.get_layers()

        from_layer_group_index, to_layer_group_index = self._convert_idxs(
            from_layer_group_index, to_layer_group_index
        )

        for layer in layers:
            if layer.layer_group_idx[0] < from_layer_group_index:
                continue
            elif layer.layer_group_idx[0] > to_layer_group_index:
                break
            else:
                if layer.is_frozen == freeze:
                    # layer already in correct state
                    continue
                else:
                    is_batch_norm = module_is_batch_norm(layer.module)
                    if is_batch_norm and not self.freeze_bn:
                        continue
                    else:
                        params = change_layer_state(
                            layer, toggle_train_eval, set_grad_value
                        )
                        if params:
                            modified_parameters[layer.layer_group_idx[0]].extend(params)

        return {layer_group_idx: {'params': params} for layer_group_idx, params in modified_parameters.items()}

    def _convert_idxs(self, from_idx, to_idx):
        from_idx = convert_idx(from_idx, self.num_groups)
        to_idx = convert_idx(to_idx, self.num_groups)

        if from_idx > to_idx:
            from_idx, to_idx = to_idx, from_idx

        return from_idx, to_idx


def change_layer_state(layer, toggle_train_eval, set_grad_value):
    params = list(layer.module.parameters())
    if params:
        set_requires_grad(params, value=set_grad_value)
    if toggle_train_eval:
        layer.module.train(mode=set_grad_value)
    return params


def module_is_batch_norm(module):
    return isinstance(module, BN_TYPES)


def convert_idx(idx, num_groups):
    if idx < 0:
        idx = idx + num_groups

    return idx


def set_requires_grad(parameters, value=True):
    for param in parameters:
        param.requires_grad = value


def get_layer_groups(module):
    layers = []
    layer_groups = dict()
    for layer_group, group in enumerate(module.children()):
        _recursive_get_layers(group, layers, layer_group)
        layer_groups[layer_group] = group

    return layer_groups, layers


def _recursive_get_layers(module, result, layer_group=0):
    children = list(module.children())
    if not children:
        # is leaf
        result.append((layer_group, module))

    else:
        # is nested
        for child in children:
            _recursive_get_layers(child, result, layer_group)


if __name__ == "__main__":
    # model = models.resnet18(pretrained=False)

    model = TestModel()

    finetuner = ModelFreezer(model)

    print("====================")
    pprint(finetuner.get_layers())
    print("------------------------")
    pprint(finetuner.get_layer_groups())
    print("====================")

    finetuner.freeze()

    print("====================")
    pprint(finetuner.get_layers())
    print("------------------------")
    # pprint(finetuner.get_layer_groups())
    print("====================")

    finetuner.unfreeze(from_index=-1, to_index=-3)
    print("====================")
    pprint(finetuner.get_layers())
    print("------------------------")
    print("====================")
    finetuner.unfreeze(from_index=-3, to_index=-4)
    print("====================")
    pprint(finetuner.get_layers())
    print("------------------------")
    print("====================")
    finetuner.unfreeze(from_index=-4, to_index=0)

    print("====================")
    pprint(finetuner.get_layers())
    print("------------------------")
    print("====================")

    # for name, param in model.named_parameters():
    #     print(name)
    #     print(param.requires_grad)
