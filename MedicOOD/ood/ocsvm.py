import torch
import torch.nn as nn
from tqdm import tqdm
from MedicOOD.dataloader.mri_datamodule import concatenate_channels
from MedicOOD.ood.ood_utils import set_forward_hooks, remove_forward_hooks, filter_activations
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from MedicOOD.ood.self_adaptive_shifting import SelfAdaptiveShifting

""""
Implement of the One-class SVM for OOD detetion
Pipeline: 
    - Attach one-class SVM to each layer activation, for novelty detection
    - Use pseudo OOD during training to fit the SVM
    - final sscore is the maximum of the individual layer scores 
Adapted from :
Layer Adaptive Deep Neural Networks for Out-of-distribution Detection https://arxiv.org/pdf/2203.00192.pdf
https://github.com/haoliangwang86/LA-OOD
"""


def getActivation(activation_dict, name):
    # Compute filter-wise average activation for layer [name]
    def hook(model, input, output):
        activ = output.detach()
        avg_activ = torch.mean(activ, dim=[2, 3, 4])
        activation_dict[name] += [avg_activ.cpu()]

    return hook


class OCSVM(nn.Module):
    """
    Class for the multi-layer One-class SVM OOD detector
    """

    def __init__(self, layer=None, verbose=False):
        super().__init__()

        self.activation_dict = {}
        self.dtype = torch.float32
        self.means = None
        self.stds = None
        self.inverse_covariances = None
        self.layer = layer
        self.verbose = verbose

    def reset(self):
        for key in self.activation_dict:
            self.activation_dict[key] = []

    def predict(self, model, x):

        batch_size = len(x)
        batch_scores = []
        hooks = set_forward_hooks(self.activation_dict, getActivation, model, verbose=False, layer_names=self.layer)
        self.reset()
        for b in range(batch_size):
            x_b = x[b, ...].unsqueeze(0)
            model(x_b)

            filter_activations(self.activation_dict)
            all_pred = []
            for key in self.activation_dict:
                test_feat = self.activation_dict[key]
                scaler = getattr(self, f'scaler_{key}')
                svm = getattr(self, f'svm_{key}')

                test_feat_scaled = scaler.transform(test_feat)
                layer_pred = - svm.decision_function(test_feat_scaled)[0]
                all_pred.append(layer_pred)

            # out_score = sum(all_pred) / len(all_pred)
            out_score = max(all_pred)
            batch_scores.append(out_score)
            self.reset()  # re-initialize dict

        # remove hooks, otherwise they are still attached to the model, which will eventually yield to a bug if
        # predict is called multiple times
        remove_forward_hooks(hooks, verbose=self.verbose)
        return batch_scores

    def fit(self, model, data_loader, device):
        """
        Fit function for OCSVM
        :param model: trained model
        :param data_loader: data loader for validation images
        :param device: GPU device
        :return:
        """
        hooks = set_forward_hooks(self.activation_dict, getActivation, model, verbose=True, layer_names=self.layer)

        for i, batch in enumerate(tqdm(data_loader, 0)):
            image_dict = concatenate_channels(batch)
            x = image_dict['image']
            x = x.to(device)

            with torch.no_grad():
                model(x)

        filter_activations(self.activation_dict, verbose=True)

        # compute orders
        for key in self.activation_dict:
            X = self.activation_dict[key].numpy()  # B, N
            ss = StandardScaler()
            ss.fit(X)
            X = ss.transform(X)

            # split dataset in train / val to find optimal parameters
            train_X, val_X = train_test_split(X, test_size=0.5, random_state=42)

            # generate pseudo-OOD using SelfAdaptiveShifting
            self_adaptive_shifting = SelfAdaptiveShifting(val_X)
            self_adaptive_shifting.edge_pattern_detection(0.01)

            pseudo_outlier_X = self_adaptive_shifting.generate_pseudo_outliers()
            pseudo_outlier_Y = -np.ones(len(pseudo_outlier_X))
            val_Y = np.ones(len(val_X))

            nu_candidates = [0.001]
            gamma_candidates = [0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]

            best_err = 1.0
            for nu in nu_candidates:
                for gamma in tqdm(gamma_candidates):
                    model = OneClassSVM(gamma=gamma, nu=nu).fit(train_X)
                    err_o = 1 - np.mean(model.predict(pseudo_outlier_X) == pseudo_outlier_Y)
                    err_t = 1 - np.mean(model.predict(val_X) == val_Y)
                    err = 0.5 * (err_o + err_t)
                    if err < best_err:
                        best_err = err
                        best_model = model

            setattr(self, f'svm_{key}', best_model)
            setattr(self, f'scaler_{key}', ss)

        remove_forward_hooks(hooks)




