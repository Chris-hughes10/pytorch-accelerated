from collections import namedtuple, defaultdict
from typing import List

import torch

BN_MODULES = (
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm3d,
    torch.nn.SyncBatchNorm,
)

Layer = namedtuple("Layer", ["layer_group_idx", "module", "is_frozen"])
LayerGroup = namedtuple("LayerGroup", ["layer_group_idx", "module", "is_frozen"])


class ModelFreezer:
    """
    A class to freeze and unfreeze different parts of a model, to simplify the process of fine-tuning during transfer learning.

    This class uses the following abstractions:
     - `Layer`: A subclass of :class:`torch.nn.Module` with a depth of 1. i.e. The module is not nested.
     - `LayerGroup`: The modules which have been defined as attributes of a model. These may be Layers or nested modules.

     For example, let's consider the following model::

        from torch import nn

        class MyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.input = nn.Linear(100, 100)
                self.block_1 = nn.Sequential(
                    nn.Linear(100, 100),
                    nn.BatchNorm1d(100),
                    nn.Sequential(
                        nn.Linear(100, 100),
                        nn.BatchNorm1d(100),
                        nn.ReLU(),
                    ),
                )
                self.output = nn.Linear(100, 10)

            def forward(self, x):
                x = self.input(x)
                x = self.block_1(x)
                out = self.output(x)
                return out

    Here, the layer groups would be the modules [`input`, `block_1`, `output`], whereas the layers would be ordered, flattened list
    of Linear, BatchNorm and ReLU modules.
    """

    def __init__(self, model, freeze_batch_norms=False):
        """
        Create a new ModelFreezer instance, which can be used to freeze and unfreeze all, or parts, or a model. When a model is passed
        to a ModelFreezer instance, all parameters will be unfrozen regardless of their previous state. Subsequent freezing/unfreezing should be
        done using this instance.

        :param model: The model to freeze/unfreeze. This should be a subclass of :class:`torch.nn.Module`
        :param freeze_batch_norms: Whether to freeze BatchNorm layers, during freezing. By default, BatchNorm layers are left unfrozen.
        """
        self.model = model
        _set_requires_grad(model.parameters(), value=True)
        self._layer_groups, self._layers = _get_layer_groups_for_module(model)
        self.freeze_bn = freeze_batch_norms
        self.num_groups = len(self._layer_groups)

    def get_layer_groups(self) -> List[LayerGroup]:
        """
        Return all of the model's layer groups. A layer group is any module which has been defined as an attribute of the model.

        :return: a list of all layer groups in the model.
        """
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

    def get_layers(self) -> List[Layer]:
        """
        Return all of the model's layers. A Layer is any non-nested module which is included in the model.

        :return: a list of all layers in the model.
        """
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

    def get_trainable_parameters(self):
        """
        Return a list of all unfrozen model parameters, which will be updated during training.

        :return: a list of all trainable parameters
        """
        return [param for param in self.model.parameters() if param.requires_grad]

    def freeze(self, from_index=0, to_index=-2, set_modules_as_eval=False):
        """
        Freeze layer groups corresponding to the specified indexes, which are inclusive. By default, this freezes all layer groups
        except the final one.

        :param from_index: The index of the first layer group to freeze.
        :param to_index: The index of the final layer group to freeze.
        :param set_modules_as_eval: If True, frozen modules will also be placed in `eval` mode. This is False by default.
        """
        self.__freeze_unfreeze(
            from_index, to_index, freeze=True, toggle_train_eval=set_modules_as_eval
        )

    def unfreeze(self, from_index=-1, to_index=0, set_modules_as_training=True):
        """
        Unfreeze layer groups corresponding to the specified indexes, which are inclusive. By default, this unfreezes all layer groups.
        For each layer group, any parameters which have been unfrozen are returned, so that they can be added to an optimizer if needed.

        :param from_index: The index of the first layer group to unfreeze.
        :param to_index: The index of the final layer group to unfreeze.
        :param set_modules_as_training: If True, unfrozen modules will also be placed in `train` mode. This is True by default.
        :return: a dictionary containing the parameters which have been unfrozen for each layer group.
        """
        unfrozen_params = self.__freeze_unfreeze(
            from_index,
            to_index,
            freeze=False,
            toggle_train_eval=set_modules_as_training,
        )
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
                    is_batch_norm = _module_is_batch_norm(layer.module)
                    if is_batch_norm and not self.freeze_bn:
                        continue
                    else:
                        params = _change_layer_state(
                            layer, toggle_train_eval, set_grad_value
                        )
                        if params:
                            modified_parameters[layer.layer_group_idx[0]].extend(params)

        return {
            layer_group_idx: {"params": params}
            for layer_group_idx, params in modified_parameters.items()
        }

    def _convert_idxs(self, from_idx, to_idx):
        from_idx = _convert_idx(from_idx, self.num_groups)
        to_idx = _convert_idx(to_idx, self.num_groups)

        if from_idx > to_idx:
            from_idx, to_idx = to_idx, from_idx

        return from_idx, to_idx


def _change_layer_state(layer: Layer, set_grad_value: bool, toggle_train_eval: bool):
    params = list(layer.module.parameters())
    if params:
        _set_requires_grad(params, value=set_grad_value)
    if toggle_train_eval:
        layer.module.train(mode=set_grad_value)
    return params


def _module_is_batch_norm(module):
    return isinstance(module, BN_MODULES)


def _convert_idx(idx, num_groups):
    if idx < 0:
        idx = idx + num_groups
    return idx


def _set_requires_grad(parameters, value=True):
    for param in parameters:
        param.requires_grad = value


def _get_layer_groups_for_module(module):
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
