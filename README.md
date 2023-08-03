# MedicOOD
Repository for the paper "Multi-layer Aggregation as a key to feature-based OOD detection" [arxiv](https://arxiv.org/pdf/2307.15647.pdf) 
![illustration](https://github.com/benolmbrt/MedicOOD/blob/master/wheel_of_ood.jpg)

This is a demonstration of the use of the 8 feature-based Out-of-distribution detectors on a toy 3D image segmentation task. 

- Step 1: Generate the toy dataset using [generate_data.py](https://github.com/benolmbert/MedicOOD/blob/master/MedicOOD/generate_data/generate_data.py)

In-distribution data: a 64x64x64 volume with spheres to segment, generated using TorchIO's RandomLabelsToImage function.
![ID](https://github.com/benolmbrt/MedicOOD/blob/master/id_data.png)

Out-of-distribution data: a ID sample is made OOD by adding TorchIO's RandomMotion artefact.
![OOD](https://github.com/benolmbrt/MedicOOD/blob/master/ood_data.png)

- Step 2: Train a simple segmentation DynUnet using [train.py](https://github.com/benolmbert/MedicOOD/blob/master/MedicOOD/model/train.py)
The training can be launched using:

 ```python MedicOOD/MedicOOD/model/train.py MedicOOD/MedicOOD/model/config.yaml```. 
 
 Don't forget to modify the paths of ```--output-folder``` and ```--data-csv``` in the YAML file.

- Step 3: Launch evaluation using [test.py](https://github.com/benolmbert/MedicOOD/blob/master/MedicOOD/model/test.py)
This script will train an instance of each feature-based OOD detector from the features of the trained DynUnet.
Then inference is launched on the test ID dataset and test OOD dataset. By comparing these scores, AUROC scores are extracted to estimate OOD detection performance.

You can use a command such as:  ```python MedicOOD/MedicOOD/model/test.py --run-folder path/to/trained/model```

| OOD detector  | AUROC |
| ------------- | ------------- |
| Spectrum Single  | 0.979  |
| Spectrum Multi  | 1.000  |
| MD Pool Single  | 0.931  |
| MD Pool Multi  | 0.979  |
| Prototypes Single  | 0.823 |
| Prototypes Multi  | 0.850 |
| FRODO  | 1.000 |
| OCSVM  | 0.987  |


- Ressources:
  
FRODO: [paper link](https://ieeexplore.ieee.org/abstract/document/9947059/)

MD Pool: [paper link](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9400372/)

Prototypes: [paper link](https://www.sciencedirect.com/science/article/abs/pii/S0950705122003410)

Spectrum: [paper link](https://arxiv.org/abs/2004.06569)

OCSVM: [paper link](https://link.springer.com/chapter/10.1007/978-3-031-05936-0_41)

- Citation:
If you use this repository in your research please cite cite us ! [arxiv](https://arxiv.org/pdf/2307.15647.pdf) 



 

