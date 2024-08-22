# DAANet

The official implementation of "Density-assisted Adaptive Alignment Network for Crowd Flux Estimation".

![image](./figures/DAANet-Overall-1.png)

# Installation

* Clone this repo in the directory ```DAANet/src/``` 
    ```bash
    cd DAANet/src
    git clone https://github.com/LeoHuang0511/DAANet.git
    ```

* Create and activate the envirnment
    ```bash
    cd DAANet/
    pip install virtualenv
    virturalenv DAANet_env
    source ./DAANet_env/bin/activate
    ```

* Install the dependencies (Python 3.8.10, PyTorch 2.1.2)
    ```bash
    pip install -r requirements.txt
    ```

# Datasets

* **SensCrowd**: Download SenseCrowd from [Baidu disk](https://pan.baidu.com/s/1OYBSPxgwvRMrr6UTStq7ZQ?pwd=64xm) or from the original dataset [link](https://github.com/HopLee6/VSCrowd-Dataset). 
* **CroHD**: Download CroHD from [here](https://motchallenge.net/data/Head_Tracking_21/). 
* **CARLA**: Download CARLA from [here](https://cycuedutw-my.sharepoint.com/:u:/g/personal/s10728241_cycu_edu_tw/EWmIb0S95hBFnb6LHWfs6hoBhxSXGdUBlR1KWqbwxP_v7w?e=q8jULl).