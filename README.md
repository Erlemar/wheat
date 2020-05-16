# wheat_detection
This is my repository with a baseline model for [Wheat Detection challenge on Kaggle](https://www.kaggle.com/c/global-wheat-detection)

Main frameworks used:
* [hydra](https://github.com/facebookresearch/hydra)
* [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning)

To use it for training, perform the following steps:
* download the data, unzip in and put in some folder;
* define that folder in config conf/data/data.yaml as a value of the key `data.folder_path`
* run run_hydra.py script

There is no script for prediction, because in this competition we have to make prediction in kernels.

Refer to my kernel for more information: https://www.kaggle.com/artgor/object-detection-with-pytorch-lightning
