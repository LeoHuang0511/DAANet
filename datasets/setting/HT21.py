from easydict import EasyDict as edict


# init
__C_HT21 = edict()

cfg_data = __C_HT21

__C_HT21.TRAIN_SIZE =(768,1024) # (848,1536) #
__C_HT21.DATA_PATH = '../../datasets/HT21/'
__C_HT21.TRAIN_LST = 'train.txt'
__C_HT21.VAL_LST =  'val.txt'
__C_HT21.TEST_LST =  'test.txt'




