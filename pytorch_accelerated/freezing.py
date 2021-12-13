import operator
from collections import namedtuple, defaultdict
from pprint import pprint
from typing import Optional, Generator

import torch
from torch import nn
from torch.optim import Optimizer

BN_TYPES = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d,
            torch.nn.LazyBatchNorm1d, torch.nn.LazyBatchNorm2d, torch.nn.LazyBatchNorm3d,
            torch.nn.SyncBatchNorm)


def filter_params(module: torch.nn.Module,
                  train_bn: bool = True) -> Generator:
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


def _unfreeze_and_add_param_group(module: torch.nn.Module,
                                  optimizer: Optimizer,
                                  lr: Optional[float] = None,
                                  train_bn: bool = True):
    """Unfreezes a module and adds its parameters to an optimizer."""
    # _make_trainable(module)
    params_lr = optimizer.param_groups[0]['lr'] if lr is None else float(lr)
    optimizer.add_param_group(
        {'params': filter_params(module=module, train_bn=train_bn),
         'lr': params_lr / 10.,
         })


def get_trainable_parameters(module: torch.nn.Module,
                  train_bn: bool = True) -> Generator:
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
            )
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

Layer = namedtuple('Layer', ['layer_group_idx', 'item',  'is_frozen'])
LayerGroup = namedtuple('LayerGroup', ['layer_group_idx', 'layers', 'is_frozen'])

class ModelFreezer:

    def __init__(self, model, freeze_batch_norms=False):
        self.model = model
        set_requires_grad(model.parameters(), value=True)
        self._layer_groups, self._layers = get_layer_groups(model)
        self.freeze_bn = freeze_batch_norms
        self.num_groups = len(self._layer_groups)

    def get_layer_groups(self):
        layer_groups = []

        for group_idx, layer_group in self._layer_groups.items():
            params = {not param.requires_grad for param in layer_group.parameters()}
            frozen = True if True in params else False

            layer_groups.append(LayerGroup((group_idx, group_idx - self.num_groups), layer_group, frozen))

        return layer_groups

    def get_layers(self):
        layers = []

        for group_idx, layer in self._layers:
            frozen_status = {param.requires_grad for param in layer.parameters(recurse=False)}
            if len(frozen_status) > 1:
                raise ValueError
            elif len(frozen_status) == 1:
                layers.append(Layer((group_idx, group_idx - self.num_groups), layer, not list(frozen_status)[0]))
            else:
                # layer has no parameters
                layers.append(Layer((group_idx, group_idx - self.num_groups), layer, not layer.training))

        return layers

    def freeze(self, to_index=-2, set_eval=True):
        self.freeze_unfreeze_to(layer_group_index=to_index, freeze=True, toggle_train_eval=set_eval)

    def unfreeze(self, to_index=0):
        #will return params that are already unfrozen
        unfrozen_params = self.freeze_unfreeze_to(layer_group_index=to_index, freeze=False)
        return unfrozen_params

    # def freeze_unfreeze_to(self, layer_group_index, freeze=True, toggle_train_eval=True):
    #     modified_parameters = defaultdict(list)
    #
    #     if layer_group_index < 0:
    #         layer_group_index = layer_group_index + self.num_groups
    #
    #     set_grad_value = not freeze
    #
    #     layers = self._layers if freeze else reversed(self._layers)
    #
    #     criterion = operator.le if freeze else operator.ge
    #
    #     for group_idx, layer in layers:
    #         if criterion(group_idx, layer_group_index):
    #             is_batch_norm = module_is_batch_norm(layer)
    #             if is_batch_norm and not self.freeze_bn:
    #                 continue
    #             else:
    #                 params = list(layer.parameters())
    #                 set_requires_grad(params, value=set_grad_value)
    #                 if toggle_train_eval:
    #                     layer.train(mode=set_grad_value)
    #                 modified_parameters[group_idx].extend(params)
    #         else:
    #             break
    #
    #     return modified_parameters

    def freeze_unfreeze_to(self, layer_group_index, freeze=True, toggle_train_eval=True):
        modified_parameters = defaultdict(list)

        if layer_group_index < 0:
            layer_group_index = layer_group_index + self.num_groups

        set_grad_value = not freeze
        layers = self.get_layers() if freeze else reversed(self.get_layers())
        criterion = operator.le if freeze else operator.ge

        for layer in layers:
            if criterion(layer.layer_group_idx[0], layer_group_index):
                if layer.is_frozen == freeze:
                    # layer already in correct state
                    continue
                else:
                    is_batch_norm = module_is_batch_norm(layer.item)
                    if is_batch_norm and not self.freeze_bn:
                        continue
                    else:
                        params = list(layer.item.parameters())
                        if params:
                            set_requires_grad(params, value=set_grad_value)
                            modified_parameters[layer.layer_group_idx[0]].extend(params)
                        if toggle_train_eval:
                            layer.item.train(mode=set_grad_value)

            else:
                break

        return modified_parameters

def module_is_batch_norm(module):
    return isinstance(module, BN_TYPES)

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


if __name__ == '__main__':
    # model = models.resnet18(pretrained=False)

    model = TestModel()

    finetuner = ModelFreezer(model)

    print('====================')
    pprint(finetuner.get_layers())
    print('------------------------')
    pprint(finetuner.get_layer_groups())
    print('====================')

    finetuner.freeze(-3)

    print('====================')
    pprint(finetuner.get_layers())
    print('------------------------')
    pprint(finetuner.get_layer_groups())
    print('====================')

    finetuner.unfreeze(-4)

    print('====================')
    pprint(finetuner.get_layers())
    print('------------------------')
    pprint(finetuner.get_layer_groups())
    print('====================')


    # for name, param in model.named_parameters():
    #     print(name)
    #     print(param.requires_grad)


