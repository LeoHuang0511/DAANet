from easydict import EasyDict as edict


# init
__C_SENSE = edict()

cfg_data = __C_SENSE

__C_SENSE.TRAIN_SIZE = (768,1024) # (848,1536) #
__C_SENSE.DATA_PATH = '../../datasets/SENSE/'
__C_SENSE.TRAIN_LST = 'train.txt'
__C_SENSE.VAL_LST =  'val.txt'
__C_SENSE.TEST_LST =  'test.txt'

