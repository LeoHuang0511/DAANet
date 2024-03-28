import datasets
# from  config import cfg
import numpy as np
import torch
import datasets
from misc.utils import *
# from model.VIC import Video_Individual_Counter
# from model.video_crowd_count import video_crowd_count
from model.video_people_flux import DutyMOFANet
from model.points_from_den import get_ROI_and_MatchInfo

from tqdm import tqdm
import torch.nn.functional as F
from pathlib import Path
import argparse
import matplotlib.cm as cm
from train import compute_metrics_single_scene,compute_metrics_all_scenes
import  os.path as osp
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
    '--GPU_ID', type=str, default='0',
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
        net = DutyMOFANet(cfg, cfg_data).cuda()

        img = torch.rand(2,3,768,1024).cuda()
        total = sum([param.nelement() for param in net.parameters()])
        print("Number of parameter: %.2fMB" % ((total/1e6)))
        macs, params = profile(net, inputs=(img,),verbose=False)
        print("Number of parameter: %.2fM" % (params/1e6))


        t = 0
        k = 100
        # _, _, _, _, _, _, _, _, _, _, _, _, _, _ = net(img)

        for i in range(k):
            start = time.time()
            _, _, _, _, _, _, _, _, _, _, _, _, _, _ = net(img)
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

# ==================== all ====================
# MAE: 12.29, MSE: 24.71  WRAE: 12.73 WIAE: 1.98 WOAE: 2.01
# ==================== in ====================
# MAE: 12.61, MSE: 23.44  WRAE: 16.94 WIAE: 1.98 WOAE: 1.94
# ==================== out ====================
# MAE: 12.19, MSE: 25.09  WRAE: 11.44 WIAE: 1.97 WOAE: 2.04
# ==================== day ====================
# MAE: 11.78, MSE: 22.94  WRAE: 12.50 WIAE: 2.05 WOAE: 2.02
# ==================== night ====================
# MAE: 14.06, MSE: 30.04  WRAE: 13.53 WIAE: 1.72 WOAE: 2.00
# ==================== scenic0 ====================
# MAE: 8.35, MSE: 17.04  WRAE: 9.61 WIAE: 1.55 WOAE: 1.80
# ==================== scenic1 ====================
# MAE: 11.15, MSE: 20.78  WRAE: 15.61 WIAE: 1.86 WOAE: 1.87
# ==================== scenic2 ====================
# MAE: 18.08, MSE: 29.65  WRAE: 17.41 WIAE: 2.29 WOAE: 2.13
# ==================== scenic3 ====================
# MAE: 33.56, MSE: 50.17  WRAE: 18.21 WIAE: 4.82 WOAE: 3.76
# ==================== scenic4 ====================
# MAE: 11.21, MSE: 21.40  WRAE: 13.06 WIAE: 1.97 WOAE: 1.69
# ==================== scenic5 ====================
# MAE: 29.91, MSE: 50.06  WRAE: 21.60 WIAE: 3.96 WOAE: 3.77
# ==================== density0 ====================
# MAE: 4.08, MSE: 5.75  WRAE: 12.58 WIAE: 1.08 WOAE: 1.21
# ==================== density1 ====================
# MAE: 7.99, MSE: 11.13  WRAE: 10.47 WIAE: 1.68 WOAE: 1.77
# ==================== density2 ====================
# MAE: 23.25, MSE: 32.89  WRAE: 14.51 WIAE: 3.14 WOAE: 3.07
# ==================== density3 ====================
# MAE: 50.03, MSE: 64.07  WRAE: 19.93 WIAE: 6.00 WOAE: 5.10
# ==================== density4 ====================
# MAE: 76.95, MSE: 83.93  WRAE: 24.46 WIAE: 6.32 WOAE: 5.99
# tensor([[ 89.4147,  93.0000],
#         [148.2149, 259.0000],
#         [ 29.7278,  37.0000],
#         [ 71.6076,  87.0000],
#         [ 38.6680,  43.0000],
#         [ 61.2983,  65.0000],
#         [ 70.3064,  77.0000],
#         [154.6082, 227.0000],
#         [ 48.5384,  50.0000],
#         [ 40.7733,  44.0000],
#         [172.4883, 284.0000],
#         [ 65.9117,  77.0000],
#         [164.5734, 232.0000],
#         [ 11.7632,  12.0000],
#         [ 41.9911,  39.0000],
#         [ 73.3622,  81.0000],
#         [ 15.7980,  22.0000],
#         [ 58.1445,  76.0000],
#         [ 78.0700,  81.0000],
#         [ 96.6829, 100.0000],
#         [123.0839, 127.0000],
#         [ 41.0895,  40.0000],
#         [ 46.0548,  43.0000],
#         [ 50.4754,  50.0000],
#         [ 54.0197,  53.0000],
#         [ 23.2172,  22.0000],
#         [ 75.5792,  69.0000],
#         [ 84.4526,  89.0000],
#         [121.0936, 115.0000],
#         [243.7008, 306.0000],
#         [ 47.3643,  45.0000],
#         [ 43.6313,  62.0000],
#         [ 77.0213,  71.0000],
#         [ 62.3260,  71.0000],
#         [ 55.6199,  54.0000],
#         [ 58.6624,  55.0000],
#         [ 37.3744,  35.0000],
#         [ 27.4079,  30.0000],
#         [ 38.2823,  41.0000],
#         [ 83.2303,  78.0000],
#         [ 26.2175,  31.0000],
#         [147.9063, 164.0000],
#         [116.8620, 160.0000],
#         [ 29.7327,  35.0000],
#         [ 27.4655,  26.0000],
#         [ 75.3924,  86.0000],
#         [157.0550, 180.0000],
#         [111.5766, 191.0000],
#         [163.5789, 171.0000],
#         [ 80.2234,  75.0000],
#         [ 88.8087, 103.0000],
#         [ 35.8673,  34.0000],
#         [107.7521, 104.0000],
#         [ 77.2214,  97.0000],
#         [ 73.8145,  70.0000],
#         [ 55.1942,  61.0000],
#         [221.1072, 237.0000],
#         [ 52.5339,  60.0000],
#         [ 64.5467,  71.0000],
#         [ 68.5728,  71.0000],
#         [ 53.1565,  47.0000],
#         [ 94.2252, 100.0000],
#         [ 57.4984,  49.0000],
#         [ 37.9518,  29.0000],
#         [ 40.8593,  33.0000],
#         [ 35.7516,  35.0000],
#         [ 80.8944,  98.0000],
#         [111.9066, 135.0000],
#         [ 30.5592,  30.0000],
#         [ 33.7159,  37.0000],
#         [ 99.1224, 101.0000],
#         [136.8730, 232.0000],
#         [ 71.0780,  69.0000],
#         [ 21.6167,  28.0000],
#         [ 46.0675,  45.0000],
#         [ 22.1183,  37.0000],
#         [ 52.8302,  54.0000],
#         [ 19.6272,  23.0000],
#         [ 35.3046,  34.0000],
#         [ 39.5665,  57.0000],
#         [ 28.2371,  16.0000],
#         [ 75.8426,  77.0000],
#         [ 26.7141,  29.0000],
#         [114.5060, 134.0000],
#         [ 72.5601,  88.0000],
#         [ 68.5220,  67.0000],
#         [112.4594, 130.0000],
#         [ 84.0589,  62.0000],
#         [227.6907, 329.0000],
#         [ 33.1534,  35.0000],
#         [ 48.8285,  47.0000],
#         [ 58.0883,  52.0000],
#         [ 57.2759,  52.0000],
#         [ 45.3223,  46.0000],
#         [ 68.1864,  64.0000],
#         [120.8597, 169.0000],
#         [ 59.3080,  64.0000],
#         [191.3976, 250.0000],
#         [175.0085, 233.0000],
#         [ 10.6638,  11.0000],
#         [ 80.5845,  97.0000],
#         [ 81.8619,  96.0000],
#         [ 35.5707,  31.0000],
#         [232.4688, 316.0000],
#         [ 70.8171,  81.0000],
#         [ 39.7070,  41.0000],
#         [ 55.2743,  57.0000],
#         [ 87.0791, 108.0000],
#         [177.9434, 190.0000],
#         [ 91.6090,  83.0000],
#         [ 51.4202,  48.0000],
#         [ 73.2659,  74.0000],
#         [ 47.3627,  70.0000],
#         [144.2767, 175.0000],
#         [ 54.6253,  63.0000],
#         [ 98.3246,  97.0000],
#         [ 56.5126,  61.0000],
#         [ 38.2883,  23.0000],
#         [ 70.7073,  69.0000],
#         [ 38.0360,  22.0000],
#         [ 53.3014,  57.0000],
#         [ 65.6217,  69.0000],
#         [ 45.7053,  53.0000],
#         [ 49.8301,  44.0000],
#         [ 23.4349,  22.0000],
#         [ 75.9271,  76.0000],
#         [ 76.7114,  76.0000],
#         [ 24.8107,  26.0000],
#         [105.8516, 116.0000],
#         [ 72.5226,  73.0000],
#         [ 68.9267,  68.0000],
#         [ 66.7460, 152.0000],
#         [ 48.6040,  56.0000],
#         [ 54.7598,  62.0000],
#         [ 65.3288,  75.0000],
#         [ 44.4746,  50.0000],
#         [158.6062, 230.0000],
#         [ 67.7573,  73.0000],
#         [ 49.4845,  37.0000],
#         [ 87.4544,  71.0000],
#         [ 86.9877,  86.0000],
#         [ 85.2368,  80.0000],
#         [ 77.9463,  75.0000],
#         [ 58.9658,  77.0000],
#         [ 51.8347,  53.0000],
#         [123.8188, 142.0000],
#         [112.5238, 164.0000],
#         [ 69.3425,  74.0000],
#         [ 62.7756,  67.0000],
#         [110.3728, 108.0000],
#         [ 50.3846,  58.0000],
#         [164.9739, 202.0000],
#         [ 26.0592,  23.0000],
#         [ 98.3755,  97.0000],
#         [ 64.1170,  62.0000],
#         [ 23.0675,  24.0000],
#         [ 48.8955,  59.0000],
#         [ 53.0321,  52.0000],
#         [ 45.3135,  43.0000],
#         [ 95.5922,  98.0000],
#         [ 46.4823,  43.0000],
#         [ 56.9812,  84.0000],
#         [172.6967, 179.0000],
#         [ 80.6760,  73.0000],
#         [ 25.7448,  25.0000],
#         [ 16.1685,  26.0000],
#         [ 62.0806,  55.0000],
#         [ 93.0958, 102.0000],
#         [ 37.5434,  51.0000],
#         [ 20.9470,  21.0000],
#         [ 42.8672,  57.0000],
#         [ 37.6022,  22.0000],
#         [ 26.5846,  26.0000],
#         [ 34.8923,  37.0000],
#         [ 42.0879,  46.0000],
#         [ 41.3744,  45.0000],
#         [116.4209, 131.0000],
#         [143.4008, 142.0000],
#         [ 28.7674,  29.0000],
#         [ 71.7286,  66.0000],
#         [ 42.6456,  69.0000],
#         [ 47.6907,  44.0000],
#         [ 26.2184,  22.0000],
#         [ 20.1989,  23.0000],
#         [ 36.0446,  36.0000],
#         [ 58.5359,  58.0000],
#         [ 17.0322,  19.0000],
#         [ 55.8627,  75.0000],
#         [126.8426, 124.0000],
#         [159.0815, 184.0000],
#         [ 37.2157,  38.0000],
#         [ 76.2001,  92.0000],
#         [106.7688, 105.0000],
#         [ 63.7852,  64.0000],
#         [247.1487, 312.0000],
#         [ 31.1691,  32.0000],
#         [ 68.5482,  67.0000],
#         [ 84.2573,  76.0000],
#         [ 64.7561,  67.0000],
#         [110.7983, 120.0000],
#         [ 51.1362,  50.0000],
#         [ 14.2311,  19.0000],
#         [ 89.2847,  87.0000],
#         [122.7155, 142.0000],
#         [ 53.4915,  54.0000],
#         [ 80.5151,  89.0000],
#         [ 16.3571,  18.0000],
#         [ 44.3656,  37.0000],
#         [118.4887, 146.0000],
#         [ 30.9687,  30.0000],
#         [ 99.9197, 108.0000],
#         [ 83.3560,  93.0000],
#         [ 40.8250,  50.0000],
#         [ 49.4679,  41.0000],
#         [264.1026, 390.0000],
#         [ 14.9821,  13.0000],
#         [ 86.6475,  91.0000],
#         [123.1816, 122.0000],
#         [ 32.7715,  36.0000],
#         [ 26.4360,  26.0000],
#         [ 49.7715,  52.0000],
#         [ 20.3532,  18.0000],
#         [ 20.4764,  18.0000],
#         [ 35.3715,  33.0000],
#         [ 75.1884,  77.0000],
#         [119.4564, 129.0000],
#         [ 47.2260,  46.0000],
#         [ 58.8956,  67.0000],
#         [ 58.2124,  69.0000],
#         [ 26.4284,  27.0000],
#         [133.6892, 137.0000],
#         [ 40.5817,  42.0000],
#         [ 32.0720,  35.0000],
#         [ 90.3207,  98.0000],
#         [ 68.1723,  77.0000],
#         [ 57.4718,  68.0000],
#         [ 21.4721,  25.0000],
#         [129.9447, 255.0000],
#         [137.8681, 112.0000],
#         [ 60.6594,  61.0000],
#         [ 26.3088,  43.0000],
#         [109.1689, 110.0000],
#         [ 77.0445,  84.0000],
#         [ 48.9617,  50.0000],
#         [ 84.1961,  81.0000],
#         [ 96.4772,  87.0000],
#         [ 56.4531,  61.0000],
#         [ 38.2997,  45.0000],
#         [ 64.8246,  63.0000],
#         [ 50.9867,  54.0000],
#         [ 26.2117,  25.0000],
#         [ 48.4944,  49.0000],
#         [ 45.4879,  79.0000],
#         [ 36.5422,  51.0000]])
