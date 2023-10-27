from easydict import EasyDict as edict


# init
__C_SENSE = edict()

cfg_data = __C_SENSE

__C_SENSE.TRAIN_SIZE = (768,1024) # (848,1536) #
__C_SENSE.DATA_PATH = '../dataset/SENSE/'
__C_SENSE.TRAIN_LST = 'train.txt'
__C_SENSE.VAL_LST =  'val.txt'
__C_SENSE.TEST_LST =  'test.txt'

# +
# __C_SENSE.MEAN_STD = (
#     [117/255., 110/255., 105/255.], [67.10/255., 65.45/255., 66.23/255.]
# )

__C_SENSE.MEAN_STD = (
    [0.5037, 0.5132, 0.5140], [0.2257, 0.0302, 0.0414]
)

# -

__C_SENSE.DEN_FACTOR = 200.

__C_SENSE.RESUME_MODEL = ''#model path
__C_SENSE.TRAIN_BATCH_SIZE = 2 #  img pairs
__C_SENSE.TRAIN_FRAME_INTERVALS=(5,25)  # 2s-5s
__C_SENSE.VAL_BATCH_SIZE = 1 # must be 1


