import numpy as np
import torch
from misc.utils import *
from model.video_crowd_flux import SOFANet

import argparse
from misc.gt_generate import *
import time
from thop import profile



parser = argparse.ArgumentParser(
    description='VIC test and demo',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--DATASET', type=str, default='SENSE',
    help='Directory where to write output frames (If None, no output)')
parser.add_argument(
    '--TASK', type=str, default='FT',
    help='Directory where to write output frames (If None, no output)')
parser.add_argument(
    '--OUTPUT_DIR', type=str, default='./test_demo',
    help='Directory where to write output frames (If None, no output)')
parser.add_argument(
    '--TEST_INTERVALS', type=int, default=11,
    help='Directory where to write output frames (If None, no output)')
parser.add_argument(
    '--SKIP_FLAG', type=bool, default=True,
    help='if you need to caculate the MIAE and MOAE, it should be False')
parser.add_argument(
    '--SAVE_FREQ', type=int, default=20,
    help='Directory where to write output frames (If None, no output)')
parser.add_argument(
    '--SEED', type=int, default=3035,
    help='Directory where to write output frames (If None, no output)')
parser.add_argument(
    '--GPU_ID', type=str, default='6',
    help='Directory where to write output frames (If None, no output)')

parser.add_argument('--VAL_BATCH_SIZE', type=int, default=1)


parser.add_argument('--TRAIN_SIZE', type=int, nargs='+', default=[768,1024])
parser.add_argument('--FEATURE_SCALE', type=float, default=1/4.)


parser.add_argument('--DEN_FACTOR', type=float, default=200.)
parser.add_argument('--MEAN_STD', type=tuple, default=([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
parser.add_argument('--ROI_RADIUS', type=float, default=4.)
parser.add_argument('--GAUSSIAN_SIGMA', type=float, default=4)
parser.add_argument('--CONF_BLOCK_SIZE', type=int, default=16)

parser.add_argument('--BACKBONE', type=str, default='vgg')



parser.add_argument(
    '--MODEL_PATH', type=str, default='',
    help='pretrained weight path')



opt = parser.parse_args()


opt.VAL_INTERVALS = opt.TEST_INTERVALS

opt.MODE = 'test'


def test(cfg, cfg_data):

    with torch.no_grad():
        net = SOFANet(cfg, cfg_data).cuda()

        img = torch.rand(2,3,768,1024).cuda()
        macs, params = profile(net, inputs=(img,),verbose=False)
        print("Number of parameter: %.2fM" % (params/1e6))


        t = 0
        k = 100

        for i in range(k):
            start = time.time()
            _, _, _, _, _, _, _, _, _, _ = net(img)
            end = time.time()
            t += end - start
        # t /= (k+1)
        e = int((k+1)) / t
        print(t)
        print("FPS: ",e)


if __name__=='__main__':
    import os
    import numpy as np
    import torch
    # from config import cfg
    from importlib import import_module
    


    # ------------prepare enviroment------------
    seed = opt.SEED
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.GPU_ID
    torch.backends.cudnn.benchmark = True

    # ------------prepare data loader------------
    data_mode = opt.DATASET
    datasetting = import_module(f'datasets.setting.{data_mode}')
    cfg_data = datasetting.cfg_data

    # ------------Start Training------------
    pwd = os.path.split(os.path.realpath(__file__))[0]
    test(opt,cfg_data)

