import torch
import torch.nn as nn
from tqdm import tqdm
from MedicOOD.dataloader.mri_datamodule import concatenate_channels
from MedicOOD.ood.ood_utils import set_forward_hooks, remove_forward_hooks, filter_activations
from torch.nn.functional import interpolate
import numpy as np

"""
Multi-layer adaption of Prototypes. 
Adapted from Diao et al. 2022, A unified uncertainty network for tumor segmentation using 
uncertainty cross entropy loss and prototype similarity
"""


def getActivation(activation_dict, name):
    # the hook signature
    def hook(model, input, output):
        activ = output.detach()
        activation_dict[name] += [activ]

    return hook


class PrototypeMulti(nn.Module):
    """
    Class to fit and infer using prototypes, extracted from feature space, One per conv layer
    """

    def __init__(self,
                 n_classes: int,
                 verbose: bool = False):
        super().__init__()

        self.n_classes = n_classes
        self.min_shape = 16   # do not use feature maps whose spatial dim is below [16, 16, 16] because interpolation fails
        self.verbose = verbose
        self.activation_dict = {}

    def reset(self):
        for key in self.activation_dict:
            self.activation_dict[key] = []

    def get_prototypes(self, features, seg):
        """
        Compute the prototype for each class using feature map and segmentation
        :param features: features of shape B M H W D
        :param seg:
        :return:
        """

        class_prototypes = []

        if features.shape[2:] != seg.shape[2:]:
            seg = torch.nn.functional.interpolate(seg.float(), size=features.shape[2:], mode='nearest')

        for n in range(self.n_classes):
            class_proto = self.get_class_prototype(features, seg, n)
            class_prototypes.append(class_proto)

        return class_prototypes

    def get_class_prototype(self, features, seg, target_class):
        """
        Compute prototype for class target_class
        :param features: features of shape B M H W D
        :param seg: segmentation of shape B 1 H W D
        :param target_class: target class id
        :return:
        """

        spatial_dims = [2, 3, 4]
        if features.shape[2:] != seg.shape[2:]:
            seg = interpolate(seg.float(), features.shape[2:], mode='nearest')  # in cases where the features has not the shape of the segmentation

        Pc = (seg == target_class).type_as(seg)
        class_cardinal = Pc.sum()
        b, m, h, w, d = features.shape
        if class_cardinal > 0:
            Ptcm = torch.sum(features * Pc, dim=spatial_dims) / class_cardinal  # eq 5 in paper --> B M
        else:
            print(f'Warning : class {target_class} is absent in image. {class_cardinal}')
            # in that case, fill the prototype with nan. It will not be taken into account when computing the
            # average prototype
            Ptcm = torch.full((b, m), torch.nan)

        return Ptcm

    def predict(self, model, x):
        """
        Compute cosine dissimilarity between the test prototypes and the ID prototypes

        :param class_prototypes: N, M
        :param features: B, M, H, D, W
        :param segmentation: B 1 H D W with N classes inside
        :return: map BHD
        """
        hooks = set_forward_hooks(self.activation_dict, getActivation, model, verbose=False, layer_names=None)
        batch_size = len(x)
        batch_oodm = []
        cosine = torch.nn.CosineEmbeddingLoss()
        self.reset()

        for b in range(batch_size):
            x_b = x[b, ...].unsqueeze(0)
            logits = model(x_b)
            segmentation = model.logits_to_segm(logits, keep_dim=True)

            filter_activations(self.activation_dict)
            layer_scores = []
            for key in self.activation_dict:  # compute the score for each layer independently
                activ = self.activation_dict[key]
                spatial_shape = np.prod(list(activ.shape[2:]))
                if spatial_shape >= self.min_shape ** 3:
                    # define test image case prototype Ptj
                    class_test_prototypes = self.get_prototypes(activ, segmentation)  # B N*M
                    valid_classes = [not torch.any(torch.isnan(prt)).item() for prt in class_test_prototypes]

                    buffered_key = f"class_prototypes_{key.replace('.', '_')}"
                    corresp_train_proto = getattr(self, buffered_key)[:, valid_classes]

                    valid_proto = []
                    for idx, status in enumerate(valid_classes):
                        if status:
                            valid_proto.append(class_test_prototypes[idx])

                    corresp_test_proto = torch.stack(valid_proto, 1)

                    flat_test_proto = torch.flatten(corresp_test_proto, start_dim=1)
                    flat_train_proto = torch.flatten(corresp_train_proto, start_dim=1).type_as(flat_test_proto)
                    targets = torch.ones(1).type_as(flat_train_proto)

                    # now compute equation 7 of paper
                    case_level_uncertainty = cosine(flat_train_proto, flat_test_proto,
                                                    target=targets).detach().cpu().item()  # b

                    layer_scores.append(case_level_uncertainty)

            # final score is the average of the scores
            avg_layer_score = sum(layer_scores) / len(layer_scores)
            batch_oodm.append(avg_layer_score)

            self.reset()

        remove_forward_hooks(hooks, verbose=self.verbose)
        return batch_oodm

    def fit(self, model, data_loader, device):
        # fit function for prototypes
        # initialize one buffer per class.
        # each list contain all the prototypes accumulated for a given class.
        # we use separate lists because it is possible than one segmentation do not contain all the classes

        self.train_prototypes = {}
        for n in range(self.n_classes):
            self.train_prototypes[n] = {}

        hooks = set_forward_hooks(self.activation_dict, getActivation, model, verbose=True, layer_names=None)
        self.valid_keys = []  # list containing the layers with spatial shape > 16
        for i, batch in enumerate(tqdm(data_loader, 0)):
            image_dict = concatenate_channels(batch)
            x = image_dict['image']
            x = x.to(device)

            with torch.no_grad():
                logits = model(x)
                seg = model.logits_to_segm(logits, keep_dim=True)
                filter_activations(self.activation_dict)
                for key in self.activation_dict:
                    activ = self.activation_dict[key]
                    spatial_shape = np.prod(list(activ.shape[2:]))
                    if spatial_shape >= self.min_shape ** 3: # keep only layer for which spatial dim is >= [8, 8, 8]
                        if key not in self.valid_keys:
                            self.valid_keys.append(key)
                        classes_proto = self.get_prototypes(activ, seg)  # list of n_classes x (b, m_features)

                        # update lists
                        for n in range(self.n_classes):
                            class_proto = classes_proto[n]
                            if not torch.any(torch.isnan(class_proto)):
                                if key not in self.train_prototypes[n]:
                                    self.train_prototypes[n][key] = []
                                self.train_prototypes[n][key].append(class_proto)

            self.reset()

        # prototype for each layer
        for key in self.valid_keys:
            # now perform aggregation, class_wise
            aggregated_prototype = []
            for n in range(self.n_classes):
                cat_class_prototypes = torch.cat(self.train_prototypes[n][key], 0)
                class_avg_prototype = torch.mean(cat_class_prototypes, 0).cpu().detach()
                aggregated_prototype.append(class_avg_prototype)

            aggregated_prototype = torch.stack(aggregated_prototype).unsqueeze(0)  # eq 6 in paper --> 1, n_classes,  m_features
            key_layer = key.replace('.', '_')
            self.register_buffer(name=f"class_prototypes_{key_layer}", tensor=aggregated_prototype)

        remove_forward_hooks(hooks)



