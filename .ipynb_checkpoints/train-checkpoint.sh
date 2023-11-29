python train_HT21.py --DATASET HT21 --EXP_NAME NoscaleWeight_MaskNoWeight_NoConfLoss_16NewConf --con_alpha 0.1 --SAVE_VIS_FREQ 500 --MAX_EPOCH 20 --GPU_ID 0  --VAL_START 999 --PRINT_FREQ 20
python train.py --DATASET HT21 --EXP_NAME NoscaleWeight_MaskNoWeight_NoConfLoss_16NewConf_val --con_alpha 0.1 --SAVE_VIS_FREQ 500 --MAX_EPOCH 20 --GPU_ID  0 --VAL_START 999 --PRINT_FREQ 20
python train.py --DATASET SENSE --EXP_NAME NoscaleWeight_MaskNoWeight_NoConfLoss_16NewConf_val --con_alpha 0.1 --SAVE_VIS_FREQ 500 --MAX_EPOCH 20 --GPU_ID  0 --VAL_START 999 --PRINT_FREQ 20
python train.py --DATASET CARLA --EXP_NAME NoscaleWeight_MaskNoWeight_NoConfLoss_16NewConf_val --con_alpha 0.1 --SAVE_VIS_FREQ 500 --MAX_EPOCH 20 --GPU_ID  0 --VAL_START 999 --PRINT_FREQ 20

