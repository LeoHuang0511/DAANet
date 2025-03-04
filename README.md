# DAANet

The official implementation of "Density-assisted Adaptive Alignment Network for Video Individual Counting".


# Installation

* Download this repo in the directory ```root/DAANet/src/``` 
    

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

* **SenseCrowd**: Download SenseCrowd from [here](https://github.com/taohan10200/DRNet).
* **CroHD**: Download CroHD from [here](https://motchallenge.net/data/Head_Tracking_21/). 
* **CARLA**: Download CARLA from [here](https://github.com/LeoHuang0511/FMDC).


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

