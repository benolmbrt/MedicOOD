import os
from argparse import ArgumentParser
import pandas as pd
from tqdm import tqdm
import torch

from MedicOOD.model.pl_model import PLModule
from MedicOOD.model.training_utils import get_best_ckpt_from_mode_and_monitor, string_to_torch_device, initialize_dm
from MedicOOD.ood.spectrum_single import SpectrumSingle
from MedicOOD.ood.spectrum_multi import SpectrumMulti
from MedicOOD.ood.md_pool_single import MDPoolSingle
from MedicOOD.ood.md_pool_multi import MDPoolMulti
from MedicOOD.ood.prototypes_single import PrototypeSingle
from MedicOOD.ood.prototypes_multi import PrototypeMulti
from MedicOOD.ood.frodo import FRODO
from MedicOOD.ood.ocsvm import OCSVM
from MedicOOD.dataloader.mri_datamodule import concatenate_channels
from sklearn.metrics import roc_auc_score

"""
Test script for the OOD detection methods. 
For each OOD detector, we perform 3 steps:
    1. Training of the OOD detector using the trained model activations, on the validation dataset
    2. Inference on test ID data
    3. Inference on test OOD data
Finally we computed classification scores (AUROC) by comparing the scores obtained on ID and OOD data.
"""


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--run-folder', type=str, required=True, help='path to the trained model folder')
    parser.add_argument('--test-id-csv', type=str, required=True, help='path to the CSV containing the paths to ID images')
    parser.add_argument('--test-ood-csv', type=str, required=True, help='path to the CSV containing the paths to OOD images')
    parser.add_argument('--device', type=int, default=None, help='GPU ID')
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--dev-run', default=False, action='store_true', help='If True, launch training and inference on 10 images')

    args = parser.parse_args()

    args.device = string_to_torch_device(args.device)
    return args


def get_ood_detector(type, target_layer):
    """
    Define the OOD detector
    :param type: type of the OOD detector
    :param target_layer: target layer for single-layer methods
    :return: the OOD detector
    """

    if type == 'spectrum_single':
        return SpectrumSingle(layer=[target_layer])
    elif type == 'spectrum_multi':
        return SpectrumMulti()
    elif type == 'md_pool_single':
        return MDPoolSingle(layer=[target_layer])
    elif type == 'md_pool_multi':
        return MDPoolMulti()
    elif type == 'prototype_single':
        return PrototypeSingle(layer=[target_layer], n_classes=2)
    elif type == 'prototype_multi':
        return PrototypeMulti(n_classes=2)
    elif type == 'frodo':
        return FRODO()
    elif type == 'ocsvm':
        return OCSVM()


def load_pl_model(folder, device):
    # load model for inference
    monitor, mode = 'dice', 'max'
    checkpoint_dir = os.path.join(folder, 'checkpoints')
    ckpt = get_best_ckpt_from_mode_and_monitor(checkpoint_dir, mode=mode, monitor=monitor)  # checkpoint with the best val dice
    model = PLModule.load_from_checkpoint(ckpt)
    model.eval()
    model.to(device)

    return model


def instantiate_dataloders(model, folder, test_id_csv, test_ood_csv, num_workers=10):
    # read validation dataset from the trained model folder
    val_df = pd.read_csv(os.path.join(folder, 'val_data.csv'))
    # read test datasets
    test_id_df = pd.read_csv(test_id_csv)
    test_ood_df = pd.read_csv(test_ood_csv)

    if args.dev_run:
        # sample 10 images for debug
        val_df = val_df.sample(10)
        test_id_df = test_id_df.sample(10)
        test_ood_df = test_ood_df.sample(10)

    # instantiate torch dataloaders
    train_dm = initialize_dm(val_df, model.hparams, num_workers=num_workers)
    train_dm.setup(stage='test')
    val_data_loader = train_dm.test_dataloader()

    test_id_dm = initialize_dm(test_id_df, model.hparams, num_workers=num_workers)
    test_id_dm.setup(stage='test')
    test_id_data_loader = test_id_dm.test_dataloader()

    test_ood_dm = initialize_dm(test_ood_df, model.hparams, num_workers=num_workers)
    test_ood_dm.setup(stage='test')
    test_ood_data_loader = test_ood_dm.test_dataloader()

    return val_data_loader, test_id_data_loader, test_ood_data_loader


if __name__ == '__main__':
    args = parse_args()
    run_folder = args.run_folder

    # this demo uses a DynUnet. We define below the name of the penultimate layer and bottleneck layer
    penultimate_conv = 'net.upsamples.3.conv_block.conv2.conv'
    bottleneck_conv = 'net.bottleneck.conv2.conv'

    # define the target layer for each method. For multi-layer methods, this is None
    target_layer = {
                    'spectrum_single': penultimate_conv,
                    'md_pool_single': bottleneck_conv,
                    'prototype_single': penultimate_conv,
                    'spectrum_multi': None,
                    'md_pool_multi': None,
                    'prototype_multi': None,
                    'frodo': None,
                    'ocsvm': None
                   }

    ood_methods = target_layer.keys()
    all_auc_scores = {}  # contain the AUROC scores for each OOD detector

    # perform evaluation for each OOD methods
    for ood_method in ood_methods:

        model = load_pl_model(run_folder, device=args.device)
        val_loader, test_id_loader, test_ood_loader = instantiate_dataloders(model, run_folder, args.test_id_csv, args.test_ood_csv)

        print(f'Train {ood_method}')
        feature_model = get_ood_detector(ood_method, target_layer[ood_method])
        feature_model.fit(model=model, data_loader=val_loader, device=args.device)

        # infer on test_id
        print(f'Test ID {ood_method}')
        test_id_scores = []
        for i, batch in enumerate(tqdm(test_id_loader, 0)):
            image_dict = concatenate_channels(batch)
            x = image_dict['image']
            x = x.to(args.device)
            score_i = feature_model.predict(model, x)
            test_id_scores += score_i

        # infer on test_id
        test_ood_scores = []
        print(f'Test OOD {ood_method}')
        for i, batch in enumerate(tqdm(test_ood_loader, 0)):
            image_dict = concatenate_channels(batch)
            x = image_dict['image']
            x = x.to(args.device)
            score_i = feature_model.predict(model, x)
            test_ood_scores += score_i

        concatenated_scores = test_id_scores + test_ood_scores
        ood_labels = [0] * len(test_id_scores) + [1] * len(test_ood_scores)
        all_auc_scores[ood_method] = roc_auc_score(ood_labels, concatenated_scores)
        print(all_auc_scores)

        del model, feature_model
        torch.cuda.empty_cache()