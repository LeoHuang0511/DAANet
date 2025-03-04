from easydict import EasyDict as edict


# init
__C_CARLA = edict()

cfg_data = __C_CARLA

__C_CARLA.DATA_PATH = '../../datasets/CARLA/'
__C_CARLA.TRAIN_LST = 'train.txt'
__C_CARLA.VAL_LST =  'val.txt'
__C_CARLA.TEST_LST =  'test.txt'



