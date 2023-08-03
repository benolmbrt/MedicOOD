import torch
import torch.nn as nn
from tqdm import tqdm
from MedicOOD.dataloader.mri_datamodule import concatenate_channels
from MedicOOD.ood.ood_utils import set_forward_hooks, remove_forward_hooks, filter_activations
import numpy as np

"""
Implementation of the single-layer MD Pool method, based on 
"Distance-based detection of out-of-distribution silent failures for Covid-19 lung lesion segmentation" from Camila Gonzalez et al. 

Pipeline: 
- Retrieve the activation of the bottleneck layer 
- Apply average pooling until the number of elements falls below of threshold (1e4 in the original paper)
- Flatten the tensors to 1D representation
- Compute gaussian distribution (mean, std, covariance)
- At test time the MD distance is computed 
"""

def getActivation(activation_dict, name, max_elem=1e4):
    """
    Retrieve the activation from layer [name] and apply average pooling until the number of elements
    falls below max_elem
    """
    # the hook signature
    def hook(model, input, output):
        activ = output.detach()
        reduced_activ = torch.clone(activ)
        nb_elem = np.prod(reduced_activ.shape)
        while nb_elem > max_elem:
            reduced_activ = torch.nn.functional.avg_pool3d(reduced_activ, kernel_size=2, stride=2)
            nb_elem = np.prod(reduced_activ.shape)

        reduced_activ = torch.flatten(reduced_activ, start_dim=1)
        activation_dict[name] += [reduced_activ.cpu()]

        del activ, reduced_activ

    return hook


class MDPoolSingle(nn.Module):
    """
    Base class for the single-layer version of the MD Pool OOD detector
    """

    def __init__(self, layer, verbose=False):
        super().__init__()

        self.activation_dict = {}
        self.dtype = torch.float32
        self.means = None
        self.stds = None
        self.inverse_covariances = None
        self.layer = layer
        assert self.layer is not None, 'layer must be provided'
        self.verbose = verbose

    def reset(self):
        for key in self.activation_dict:
            self.activation_dict[key] = []

    def predict(self, model, x):
        """
        Compute MD score given a trained model and input image x
        :param model: trained segmentation model
        :param x: input image
        :return:
        """

        batch_size = len(x)
        batch_scores = []
        hooks = set_forward_hooks(self.activation_dict, getActivation, model, verbose=self.verbose, layer_names=self.layer)
        self.reset()

        with torch.no_grad():
            for b in range(batch_size):
                x_b = x[b, ...].unsqueeze(0)
                model(x_b)
                filter_activations(self.activation_dict)

                total_ood_score = []
                # compute signature score for each layer
                for key in self.activation_dict:
                    # signatures
                    mean_train_activ = getattr(self, f"mean_activ_{key.replace('.', '_')}")
                    std_train_activ = getattr(self, f"std_activ_{key.replace('.', '_')}")
                    inv_cov_train_activ = getattr(self, f"inv_cov_activ_{key.replace('.', '_')}")
                    test_activ = self.activation_dict[key].detach().cpu()

                    norm_test_activ = (test_activ - mean_train_activ) / std_train_activ
                    norm_test_activ = norm_test_activ.unsqueeze(-1)

                    md = torch.matmul(norm_test_activ.mT, torch.matmul(inv_cov_train_activ, norm_test_activ)).squeeze()
                    md = torch.sqrt(md).item()
                    total_ood_score += [md]

                avg_layer_score = sum(total_ood_score) / len(total_ood_score)

                batch_scores.append(avg_layer_score)
                self.reset()

        # remove hooks, otherwise they are still attached to the model, which will eventually yield to a bug if
        # predict is called multiple times
        remove_forward_hooks(hooks, verbose=self.verbose)

        return batch_scores

    def fit(self, model, data_loader, device):
        hooks = set_forward_hooks(self.activation_dict, getActivation, model, verbose=True, layer_names=self.layer)

        for i, batch in enumerate(tqdm(data_loader, 0)):
            image_dict = concatenate_channels(batch)
            x = image_dict['image']
            x = x.to(device)

            with torch.no_grad():
                model(x)

        filter_activations(self.activation_dict)

        for key in self.activation_dict:
            # signatures
            activ = self.activation_dict[key]
            mean_activ = torch.mean(activ, 0)
            std_activ = torch.std(activ, 0)
            std_zeros = (std_activ == 0)
            std_activ = std_activ + std_zeros * torch.ones_like(std_activ)

            norm_activ = (activ - mean_activ) / std_activ

            covariances = torch.cov(norm_activ.T)
            I = torch.eye(mean_activ.shape[0], dtype=self.dtype)
            epsilon = torch.tensor([0.01], dtype=self.dtype)

            covariances = covariances + (I * epsilon)
            inverse_covariances = torch.linalg.inv(covariances)

            key_formatted = key.replace('.', '_')
            self.register_buffer(f'mean_activ_{key_formatted}', mean_activ)
            self.register_buffer(f'std_activ_{key_formatted}', std_activ)
            self.register_buffer(f'inv_cov_activ_{key_formatted}', inverse_covariances)

        remove_forward_hooks(hooks)










