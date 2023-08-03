import torch
import torch.nn as nn
from tqdm import tqdm
from MedicOOD.dataloader.mri_datamodule import concatenate_channels
from MedicOOD.ood.ood_utils import set_forward_hooks, remove_forward_hooks, filter_activations

"""
Multi-layer adaptation of Spectrum Single. 
Adapted from Karimi et al. 2020, Improving Calibration and Out-of-Distribution Detection in Medical Image Segmentation 
with Convolutional Neural Networks

Here we do so for each layer 
"""


def getActivation(activation_dict, name):
    # the hook signature
    def hook(model, input, output):
        activ = output.detach()
        activation_dict[name] += [activ]

    return hook


class SpectrumMulti(nn.Module):
    """
    Class to fit and infer using spectrum of feature maps / one per conv
    """

    def __init__(self,
                 verbose: bool = False):
        super().__init__()

        self.verbose = verbose
        self.activation_dict = {}

    def reset(self):
        for key in self.activation_dict:
            self.activation_dict[key] = []

    def predict(self, model, x):
        # Compute the distance between test signature and ID signatures for each layer, then compute average

        hooks = set_forward_hooks(self.activation_dict, getActivation, model, verbose=False)
        batch_size = len(x)
        batch_oodm = []
        self.reset()
        for b in range(batch_size):
            x_b = x[b, ...].unsqueeze(0)
            model(x_b)

            filter_activations(self.activation_dict)
            all_distances = []
            for key in self.activation_dict:
                test_activ = self.activation_dict[key]
                test_signature = self.get_spectrum(test_activ) # 1 X, X
                buffered_key = f"all_signatures_{key.replace('.', '_')}"
                buffered_activ = getattr(self, buffered_key)
                # compute Out-Of-Distribution-Measure (OODM) (eq 1 in paper)
                distances = torch.sqrt(torch.pow(test_signature[0] - buffered_activ.type_as(test_signature), 2).sum(dim=[1, 2]))
                OODM = torch.min(distances).detach().cpu().item()
                all_distances.append(OODM)

            avg_ood_score = sum(all_distances) / len(all_distances)
            batch_oodm.append(avg_ood_score)
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
        reshape_features = torch.flatten(features, start_dim=2)  # b, n, h*w*d
        reshape_features = torch.swapaxes(reshape_features, 1, -1)  # b, h*w*d, n

        batch_signatures = []
        for batch_item in range(b):
            feat_b = reshape_features[batch_item]  # h*d*w, n
            S = torch.linalg.svdvals(feat_b)
            spectrum_b = torch.log(torch.diag(S) + 1e-10)
            l2_norm = torch.linalg.norm(spectrum_b)
            spectrum_b /= l2_norm  # spectral signature = normalized logarithmic spectrum of feature maps
            batch_signatures.append(spectrum_b)

        batch_signatures = torch.stack(batch_signatures, dim=0)
        return batch_signatures


    def fit(self, model, data_loader, device):
        hooks = set_forward_hooks(self.activation_dict, getActivation, model, verbose=True)

        all_signatures = {}  # list containing the signature for each image in data_loader
        for i, batch in enumerate(tqdm(data_loader, 0)):
            image_dict = concatenate_channels(batch)
            x = image_dict['image']
            x = x.to(device)

            with torch.no_grad():
                model(x)
                filter_activations(self.activation_dict, verbose=self.verbose)

                for key in self.activation_dict:
                    if key not in all_signatures:
                        all_signatures[key] = []
                    activations = self.activation_dict[key]
                    types = self.get_spectrum(activations)  # b, k

                    all_signatures[key].append(types)

                self.reset()

        for key in self.activation_dict:
            key_sig = torch.concat(all_signatures[key], 0).detach().cpu()  # len(dataloader), k
            saved_key = key.replace('.', '_')
            self.register_buffer(name=f"all_signatures_{saved_key}", tensor=key_sig)