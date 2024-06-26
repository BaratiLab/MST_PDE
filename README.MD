# MST_PDE

This directory contains the code for the paper [Multi-Scale Time-Stepping of PDEs with Transformers](https://www.sciencedirect.com/science/article/pii/S0045782524002391).

## Installation

To use the MST_PDE framework, follow these steps:

1. Clone the repository: 

`git clone https://github.com/BaratiLab/MST_PDE.git`

2. Create a new environment and install the required dependencies: 

`conda create -n MSE_PDE`

`conda activate MST_PDE`

`pip install -r requirements.txt`

## Datasets:

We use the NS datasets available [here](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-?usp=drive_link)(excluding the r256 file with v1e-4). Please use the code `data_cleaner.py` to pre-process the data after downloading and extracting it from the google drive.

## Usage

To train the autoencoder for the NS datasets:

`bash Experiments/train_NS_AE.sh`

Then, to train the multi-scale models for the NS datasets:

`bash Experiments/train_NS_D.sh`

Finally, to test the trained models:

`bash Experiments/test_NS.sh`

To train both AE and D (only D1 worked) for KF:

`bash Experiments/train_KF_AE_D.sh`

And to test the KF model:

`bash Experiments/test_KF.sh`

To run the ablation study on training rollout, number of time-scales and positional encoding mechanism:

`bash Experiments/train_NS_ablation.sh`

`bash Experiments/test_NS_ablation.sh`

You need to have trained the autoencoders for these. They reuse them for the ablation study.

The results for each model will be saved in the corresponding folder under the folder `Results`.

