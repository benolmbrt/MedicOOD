import torch
import torch.nn as nn
import functools

"""
List of utils for OOD score computation
"""


def rgetattr(obj, attr, *args):
    """
    Recursively get an attribute in an objected. Use to access a given layer in a torch model
    """

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))


def filter_activations(activation_dict, verbose=False):
    """
    Remove undesired activations from activation_dict. In particular, we remove the keys for which no activations
    were collected (nb_of_samples == 0)
    :param activation_dict:
    :param verbose:
    :return:
    """
    keys_to_remove = []
    for key in activation_dict.keys():
        nb_of_samples = len(activation_dict[key])
        if nb_of_samples == 0:
            keys_to_remove.append(key)
            if verbose:
                print(f'Remove layer {key} because no activations were found.')
        else:
            activation_dict[key] = torch.concat(activation_dict[key], dim=0)

    for key2remove in keys_to_remove:
        activation_dict.pop(key2remove)


def set_forward_hooks(activation_dict, registered_function, net, verbose=False, layer_names=None):
    """
    Put forward hooks to gather activations, using the name layer_names. If layer_names is not provided, we automatically
    detect all convolution layers in the networks and put a hook on all layers
    :param activation_dict: dict containing all the activations
    :param registered_function: function that collects the activation of the layer
    :param model: trained segmentation model
    :param verbose: if verbose, print the name of the layers that have been hooked
    :param layer_names: if None, put a hook on each conv layer. Else, this is a list containing the name of the layers
    :return:
    """
    if layer_names is None:
        # find all convolutions
        layer_names = [name for name, module in net.named_modules() if type(module) in [nn.Conv2d, nn.Conv3d]]
        layer_names.remove('net.output_block.conv.conv')  # remove output conv (for the DynUnet only)

    else:
        assert isinstance(layer_names, list), 'If layer_names is provided, expect a list of layer names'
        for l in layer_names:
            try:
                # look for the layer called "l" in the net
                rgetattr(net, l)
            except:
                available_layers = [name for name, module in net.named_modules() if type(module) in [nn.Conv2d, nn.Conv3d]]  # list of layer names
                raise ValueError(f'layer {l} not found. The available convolution layers are {available_layers}')

    if verbose:
        print('Setting hooks on the following conv layers : {}'.format(layer_names))

    hook_list = []
    for i, layer in enumerate(layer_names):
        # for each layer in layer_names, register a forward hook that gather the activation of the layer
        conv_layer = rgetattr(net, layer)
        hook = conv_layer.register_forward_hook(registered_function(activation_dict, layer))
        activation_dict[layer] = []
        hook_list.append(hook)

    return hook_list


def remove_forward_hooks(hooks, verbose=False):
    """
    Remove the hooks
    :param hooks: list of hooks
    :param verbose:
    :return:
    """
    if verbose:
        nb_hooks = len(hooks)
        print(f'Removing {nb_hooks} hooks')
    for hook in hooks:
        hook.remove()

