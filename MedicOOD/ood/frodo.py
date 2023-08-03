import torch
import torch.nn as nn
from tqdm import tqdm
from MedicOOD.dataloader.mri_datamodule import concatenate_channels
from MedicOOD.ood.ood_utils import set_forward_hooks, remove_forward_hooks, filter_activations

"""
PyTorch implementation of the FRODO algorithm from: 
- "FRODO: An in-depth analysis of a system to reject outlier samples from a trained neural network" and 
- "FRODO: Free rejection of out-of-distribution samples in medical imaging" from Erdi Çallı et al.

This implementation is based on the TensorFlow original implementation : https://github.com/DIAGNijmegen/frodo

Pipeline: 
- Retrieve the activation of all convolutions and compute the average over spatial dimensions  
- Compute mutlivariate gaussian distributions (mean, std, covariance) for each layer 
- At test time the MD distance is computed for each layer, then a global normalized score is computed 
"""


def getActivation(activation_dict, name):
    # Compute filter-wise average activation for layer [name]
    def hook(model, input, output):
        activ = output.detach()
        avg_activ = torch.mean(activ, dim=[2, 3, 4])  # mean over spatial dims
        activation_dict[name] += [avg_activ.cpu()]

    return hook


class FRODO(nn.Module):
    """
    Base class for the FRODO OOD detector
    """

    def __init__(self, layer=None, verbose=False):
        super().__init__()

        self.activation_dict = {}  # contain the activations for all hooked layers
        self.dtype = torch.float32
        self.means = None  # fitted means for each layer
        self.stds = None  # fitted stds for each layer
        self.inverse_covariances = None  # fitted inverse covariances for each layer
        self.layer = layer
        self.verbose = verbose

    def reset(self):
        # reset the activations
        for key in self.activation_dict:
            self.activation_dict[key] = []

    def predict(self, model, x):
        """
        Compute FRODO score given a trained model and input image x
        :param model: trained segmentation model
        :param x: input image
        :return:
        """

        batch_size = len(x)
        batch_scores = []
        hooks = set_forward_hooks(self.activation_dict, getActivation, model, verbose=False, layer_names=self.layer)
        self.reset()
        for b in range(batch_size):
            x_b = x[b, ...].unsqueeze(0)
            model(x_b)

            filter_activations(self.activation_dict)
            unormalized_preds = [self.activation_dict[key] for key in self.activation_dict]
            normalized_preds = [((output - mean) / std)
                                  for mean, std, output in zip(self.means, self.stds, unormalized_preds)]

            num_features = [output.shape[-1] for output in normalized_preds]
            normalized_preds = [torch.unsqueeze(normalized_pred, -1) for normalized_pred in normalized_preds]

            mahalanobis_distances = [torch.matmul(norm_out.mT, torch.matmul(inv_cov, norm_out)) for norm_out, inv_cov in
                                     zip(normalized_preds, self.inverse_covariances)]

            mahalanobis_distances = [torch.squeeze(md) for md in mahalanobis_distances]
            mahalanobis_distances = [torch.sqrt(md) for md in mahalanobis_distances]

            globally_normalized_mahalanobis_distances = torch.stack([md / num_feature
                                                         for md, num_feature in zip(mahalanobis_distances, num_features)])

            frodo = torch.mean(globally_normalized_mahalanobis_distances, axis=0).item()

            batch_scores.append(frodo)
            self.reset()  # re-initialize dict for the next image

        # remove hooks, otherwise they are still attached to the model
        remove_forward_hooks(hooks, verbose=self.verbose)
        return batch_scores

    def fit(self, model, data_loader, device):
        """
        Fit function for FRODO
        :param model: trained segmentation model
        :param data_loader: data loader with validation images
        :param device: GPU device
        :return:
        """

        # put the forward hooks on the convolution layers
        hooks = set_forward_hooks(self.activation_dict, getActivation, model, verbose=True, layer_names=self.layer)

        for i, batch in enumerate(tqdm(data_loader, 0)):
            image_dict = concatenate_channels(batch)
            x = image_dict['image']
            x = x.to(device)

            with torch.no_grad():
                # forward to collect activ
                model(x)

        filter_activations(self.activation_dict)

        # compute orders
        means = [torch.mean(self.activation_dict[key], axis=0, keepdims=False) for key in self.activation_dict]
        stds = [torch.std(self.activation_dict[key], axis=0) for key in self.activation_dict]
        unormalized_features = [self.activation_dict[key] for key in self.activation_dict]

        zero_stds = [std == 0 for std in stds]

        stds = [std + zero_std * torch.ones_like(std, dtype=self.dtype) for std, zero_std in zip(stds, zero_stds)]

        normalized_features = [(features - mean) / std for mean, std, features in
                               zip(means, stds, unormalized_features)]

        covariances = [torch.cov(features.T) for features in normalized_features]
        Is = [torch.eye(mean.shape[0], dtype=self.dtype) for mean in means]
        epsilon = torch.tensor([0.01], dtype=self.dtype)

        covariances = [covariance + (I * epsilon) for covariance, I in zip(covariances, Is)]
        inverse_covariances = [torch.linalg.inv(covariance) for covariance in covariances]

        self.means = means
        self.stds = stds
        self.inverse_covariances = inverse_covariances

        remove_forward_hooks(hooks)




