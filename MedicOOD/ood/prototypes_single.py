import torch
import torch.nn as nn
from tqdm import tqdm
from MedicOOD.dataloader.mri_datamodule import concatenate_channels
from MedicOOD.ood.ood_utils import set_forward_hooks, remove_forward_hooks, filter_activations
from torch.nn.functional import interpolate

"""
Feature-based OOD detection based on class prototypes. Post-Hoc.
From Diao et al. 2022, A unified uncertainty network for tumor segmentation using 
uncertainty cross entropy loss and prototype similarity
"""


def getActivation(activation_dict, name):
    # the hook signature
    def hook(model, input, output):
        activ = output.detach()
        activation_dict[name] += [activ]

    return hook


class PrototypeSingle(nn.Module):
    """
    Class to fit and infer using prototypes, extracted from feature space.
    """

    def __init__(self,
                 n_classes: int,
                 layer: str,
                 verbose: bool = False):
        super().__init__()

        self.n_classes = n_classes
        self.layer = layer
        assert len(self.layer) == 1, f'Prototypes expect a single target layer but got {self.layer}'
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
            seg = interpolate(seg.float(), features.shape[2:], mode='nearest')

        Pc = (seg == target_class).type_as(seg)  # binarize segmentation
        class_cardinal = Pc.sum()
        b, m, h, w, d = features.shape
        if class_cardinal > 0:
            Ptcm = torch.sum(features * Pc, dim=spatial_dims) / class_cardinal  # eq 5 in paper --> B M
        else:
            print(f'Warning : class {target_class} is absent in image.')
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
        hooks = set_forward_hooks(self.activation_dict, getActivation, model, verbose=False, layer_names=self.layer)
        batch_size = len(x)
        batch_oodm = []
        cosine = torch.nn.CosineEmbeddingLoss()
        self.reset()

        for b in range(batch_size):
            x_b = x[b, ...].unsqueeze(0)
            logits = model(x_b)
            segmentation = model.logits_to_segm(logits, keep_dim=True)

            filter_activations(self.activation_dict)
            activ = self.activation_dict[self.layer[0]]
            b, _, h, d, w = activ.shape

            # define test image case prototype Ptj
            class_test_prototypes = self.get_prototypes(activ, segmentation)  # B N*M
            valid_classes = [not torch.any(torch.isnan(prt)).item() for prt in class_test_prototypes]
            corresp_train_proto = self.class_prototypes[:, valid_classes]

            # some classes may be absent from the test segmentation. In that case the prototype reduces to the classes
            # present in the mask
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

            batch_oodm.append(case_level_uncertainty)
            self.reset()

        remove_forward_hooks(hooks, verbose=self.verbose)
        return batch_oodm

    def fit(self, model, data_loader, device):
        # fit function for prototypes
        # initialize one buffer per class.
        # each list contain all the prototypes accumulated for a given class.
        # we use separate lists because it is possible than one segmentation do not contain all the classes

        # initialize the lists
        self.train_prototypes = {}
        for n in range(self.n_classes):
            self.train_prototypes[n] = []

        hooks = set_forward_hooks(self.activation_dict, getActivation, model, verbose=True, layer_names=self.layer)

        for i, batch in enumerate(tqdm(data_loader, 0)):
            image_dict = concatenate_channels(batch)
            x = image_dict['image']
            x = x.to(device)

            with torch.no_grad():
                logits = model(x)
                seg = model.logits_to_segm(logits, keep_dim=True)
                filter_activations(self.activation_dict)
                activ = self.activation_dict[self.layer[0]]
                classes_proto = self.get_prototypes(activ, seg)  # list of n_classes x (batch_size, m_features)

            # update lists
            for n in range(self.n_classes):
                class_proto = classes_proto[n]
                if not torch.any(torch.isnan(class_proto)):
                    self.train_prototypes[n].append(class_proto)

            self.reset()

        # now perform aggregation, class_wise
        aggregated_prototype = []
        for n in range(self.n_classes):
            cat_class_prototypes = torch.cat(self.train_prototypes[n], 0)
            class_avg_prototype = torch.mean(cat_class_prototypes, 0).cpu().detach()
            aggregated_prototype.append(class_avg_prototype)

        aggregated_prototype = torch.stack(aggregated_prototype).unsqueeze(
            0)  # eq 6 in paper --> 1, n_classes,  m_features
        self.register_buffer(name="class_prototypes", tensor=aggregated_prototype)

        remove_forward_hooks(hooks)



