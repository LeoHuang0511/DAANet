# DAANet

The official implementation of "Density-assisted Adaptive Alignment Network for Video Individual Counting".


# Installation

* Clone this repo in the directory ```root/DAANet/src/``` 
    ```bash
    cd root/DAANet/src
    git clone https://github.com/LeoHuang0511/DAANet.git
    ```

* Create and activate the envirnment
    ```bash
    cd root/DAANet/
    pip install virtualenv
    virturalenv DAANet_env
    source ./DAANet_env/bin/activate
    ```

* Install the dependencies (Python 3.8.10, PyTorch 2.1.2)
    ```bash
    pip install -r requirements.txt
    ```

# Datasets

* **SenseCrowd**: 
    1. Download SenseCrowd from [Baidu disk](https://pan.baidu.com/s/1OYBSPxgwvRMrr6UTStq7ZQ?pwd=64xm).
    2. Download the original dataset form [here](https://github.com/HopLee6/VSCrowd-Dataset) and the lists of `train/val/test` sets at [link1](https://1drv.ms/u/s!AgKz_E1uf260nWeqa86-o9FMIqMt?e=0scDuw) or [link2](https://pan.baidu.com/s/13X3-egn0fYSd6NUTxB4cuw?pwd=ew8f), and place them to each dataset folder, respectively.  
* **CroHD**: Download CroHD from [here](https://motchallenge.net/data/Head_Tracking_21/). 
* **CARLA**: Download CARLA from [here](https://drive.google.com/file/d/1hycxlqE66QGXsOo-HMWyi3WUlF2TII8r/view?usp=sharing).

# Pretrained Weights

* The pretrained weights of our model are available at [drive](https://drive.google.com/drive/folders/1XBYG-cpNwLZZKDa2acr2qp0nLFLXrxpO?usp=drive_link).

# Preparation

Put the downloaded datasets in the directory ```root/datasets/```, forming the folder structure like:
```
root
├──DAANet
│   └──src
├──exp
│   └──pretrained
│       ├──SensCrowd.pth
│       ├──CARLA.pth
│       └──HT21.pth
└──datasets
    ├──Sense
    ├──CARLA
    └──HT21
     
```

# Training

* Run the following command to train your own model:
    ```bash
    python train.py --DATASET <dataset_name> --GPU_ID 0
    ```
* The checkpoints would be saved in ```root/DAANet/exp/```.

# Testing

* Run the following command to test the model pretrained on SenseCrowd:
    ```bash
    python test_<dataset_name>.py --MODEL_PATH <pretrained_weights_path> --GPU_ID 0
    ```

