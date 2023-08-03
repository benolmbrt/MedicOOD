import torch
import torch.nn as nn
from tqdm import tqdm
from MedicOOD.dataloader.mri_datamodule import concatenate_channels
from MedicOOD.ood.ood_utils import set_forward_hooks, remove_forward_hooks, filter_activations

"""
Feature-based OOD detection based on spectral signature in features space. Post-Hoc.
From Karimi et al. 2020, Improving Calibration and Out-of-Distribution Detection in Medical Image Segmentation with 
Convolutional Neural Networks
"""


def getActivation(activation_dict, name):
    # the hook signature
    def hook(model, input, output):
        activ = output.detach()
        activation_dict[name] += [activ]

    return hook


class SpectrumSingle(nn.Module):
    """
    Class to fit and infer using spectrum of feature maps.
    """

    def __init__(self,
                 layer: str,
                 verbose: bool = False):
        super().__init__()

        self.layer = layer
        assert len(self.layer) == 1, 'SpectrumSingle works with a single layer'
        self.verbose = verbose
        self.activation_dict = {}

    def reset(self):
        for key in self.activation_dict:
            self.activation_dict[key] = []

    def predict(self, model, x):
        # Compute the distance between test signature and ID signatures
        hooks = set_forward_hooks(self.activation_dict, getActivation, model, verbose=False, layer_names=self.layer)
        batch_size = len(x)
        batch_oodm = []
        self.reset()
        for b in range(batch_size):
            x_b = x[b, ...].unsqueeze(0)
            model.forward(x_b)

            filter_activations(self.activation_dict)
            test_activ = self.activation_dict[self.layer[0]]
            test_signature = self.get_spectrum(test_activ) # 1 X, X
            # compute Out-Of-Distribution-Measure (OODM) (eq 1 in paper)
            distances = torch.sqrt(torch.pow(test_signature[0] - self.all_signatures.type_as(test_signature), 2).sum(dim=[1, 2]))
            OODM = torch.min(distances).detach().cpu().item()

            batch_oodm.append(OODM)
            self.reset()

        remove_forward_hooks(hooks, verbose=self.verbose)
        return batch_oodm

    def get_spectrum(self, features):
        """
        Compute spectrum for a given feature maps
        :param features: features B M H W  D
        :return:
        """
        b, n, h, w, d = features.shape
        # to 2D representation (+ batch)
        reshape_features = torch.flatten(features, start_dim=2)  # b, n, h*w*d
        reshape_features = torch.swapaxes(reshape_features, 1, -1)  # b, h*w*d, n

        batch_signatures = []
        for batch_item in range(b):
            # spectral signature is based on Singular Value Decomposition
            feat_b = reshape_features[batch_item]  # h*d*w, n
            S = torch.linalg.svdvals(feat_b)
            spectrum_b = torch.log(torch.diag(S) + 1e-10)
            l2_norm = torch.linalg.norm(spectrum_b)
            spectrum_b /= l2_norm  # spectral signature = normalized logarithmic spectrum of feature maps
            batch_signatures.append(spectrum_b)

        batch_signatures = torch.stack(batch_signatures, dim=0)
        return batch_signatures


    def fit(self, model, data_loader, device):
        hooks = set_forward_hooks(self.activation_dict, getActivation, model, verbose=True, layer_names=self.layer)

        all_signatures = []  # list containing the signature for each image in data_loader
        for i, batch in enumerate(tqdm(data_loader, 0)):
            image_dict = concatenate_channels(batch)
            x = image_dict['image']
            x = x.to(device)

            with torch.no_grad():
                model.forward(x)
                filter_activations(self.activation_dict, verbose=self.verbose)

                activations = self.activation_dict[self.layer[0]]
                types = self.get_spectrum(activations)  # b, k
                all_signatures.append(types)
                self.reset()

        all_signatures = torch.concat(all_signatures, 0).detach().cpu()  # len(dataloader), k
        self.register_buffer(name="all_signatures", tensor=all_signatures)