#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from PIL import Image, ImageFont, ImageDraw

import os
import numpy as np
import torch
# from config import cfg
from importlib import import_module
import misc.transforms as own_transforms




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
    '--OUTPUT_DIR', type=str, default='./visualization',
    help='Directory where to write output frames (If None, no output)')
parser.add_argument(
    '--TEST_INTERVALS', type=int, default=11,
    help='Directory where to write output frames (If None, no output)')
parser.add_argument(
    '--SKIP_FLAG', type=bool, default=True,
    help='if you need to caculate the MIAE and MOAE, it should be False')
parser.add_argument(
    '--SAVE_FREQ', type=int, default=200,
    help='Directory where to write output frames (If None, no output)')
parser.add_argument(
    '--SEED', type=int, default=3035,
    help='Directory where to write output frames (If None, no output)')
parser.add_argument(
    '--GPU_ID', type=str, default='1',
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

# parser.add_argument(
#     '--MODEL_PATH', type=str, default='./exp/SENSE/03-22_17-33_SENSE_VGG16_FPN_5e-05/ep_15_iter_115000_mae_2.gaussian_kernel1_mse_3.677_seq_MAE_6.439_WRAE_9.506_MIAE_1.447_MOAE_1.474.pth',
#     help='pretrained weight path')


opt = parser.parse_known_args()[0]

# opt = parser.parse_args()


opt.VAL_INTERVALS = opt.TEST_INTERVALS

opt.MODE = 'vis'




# In[2]:


from collections import defaultdict
# def Target(root,i,frame):
#     img_ids = os.listdir(root)
#     img_ids.sort()
#     labels=[]
#     gts = defaultdict(list)
#     with open(root.replace('video_ori', 'label_list_all')+'.txt', 'r') as f: #label_list_all_rmInvalid
#         lines = f.readlines()
#         for lin in lines:
#             lin_list = [i for i in lin.rstrip().split(' ')]
#             ind = lin_list[0]
#             lin_list = [float(i) for i in lin_list[3:] if i != '']
#             assert len(lin_list) % 7 == 0
#             gts[ind] = lin_list
        
#     img_id = frame.strip()
#     # single_path = osp.path.join(root, img_id)
#     label = gts[img_id]
#     box_and_point = torch.tensor(label).view(-1, 7).contiguous()

#     points = box_and_point[:, 4:6].float()
#     ids = (box_and_point[:, 6]).long()

#     if ids.size(0)>0:
#         sigma = 0.6*torch.stack([(box_and_point[:,2]-box_and_point[:,0])/2,(box_and_point[:,3]-box_and_point[:,1])/2],1).min(1)[0]  #torch.sqrt(((box_and_point[:,2]-box_and_point[:,0])/2)**2 + ((box_and_point[:,3]-box_and_point[:,1])/2)**2)
#     else:
#         sigma = torch.tensor([])

#     labels.append({'scene_name':i,'frame':int(img_id.split('.')[0].replace('_resize','')), 'person_id':ids, 'points':points, 'sigma':sigma})

#     return labels

def Target(base_path,i,frame):
    # img_path = []
    labels=[]
    root  = osp.join(base_path,'img1')
    img_ids = os.listdir(root)
    img_ids.sort()
    gts = defaultdict(list)
    with open(osp.join(root.replace('img1', 'gt'), 'gt.txt'), 'r') as f:
        lines = f.readlines()
        for lin in lines:
            lin_list = [float(i) for i in lin.rstrip().split(',')]
            ind = int(lin_list[0])
            gts[ind].append(lin_list)
    img_id = frame.strip()
    # print(img_id)
    # print(gts)
    # single_path = osp.join(root, img_id)
    # print(len(gts))
    # print(gts[1])
    annotation  = gts[int(img_id.split('.')[0].replace('_resize',''))]
    annotation = torch.tensor(annotation,dtype=torch.float32)
    # print(annotation)
    box = annotation[:,2:6]
    points =   box[:,0:2] + box[:,2:4]/2

    sigma = torch.min(box[:,2:4], 1)[0] / 2.
    ids = annotation[:,1].long()
    # img_path.append(single_path)

    labels.append({'scene_name':i,'frame':int(img_id.split('.')[0].replace('_resize','')), 'person_id':ids, 'points':points,'sigma':sigma})
    return labels


# In[3]:


opt.MODEL_PATH = "/nfs/home/leo0511/Research/DutyMOFA/exp/CARLA/02-22_15-57_vgg_Crop0613_CARLA_1e-05_0.0001/latest_state.pth"
scene_name = "02" #"1019_IMG_1639_cut_01"
index = 40
while (index+343 < 410):
    # try:
    print(index)

    img1_frame = '{:0>6}'.format(343)
    img2_frame =  '{:0>6}'.format(index+343)
    img_1_path = f"/nfs/home/leo0511/Research/datasets/CARLA/train/{scene_name}"
    # img_2_path = f"/nfs/home/leo0511/Research/datasets/SENSE/video_ori/{scene_name}"
    img_2_path = f"/nfs/home/leo0511/Research/datasets/CARLA/train/{scene_name}"

    target1 = Target(img_1_path,scene_name,img1_frame+'.jpg')
    target2 = Target(img_2_path,scene_name,img2_frame+'.jpg')
    # img1 = Image.open(os.path.join(img_1_path,img1_frame+'.jpg'))
    # img2 = Image.open(os.path.join(img_2_path,img2_frame+'.jpg'))
    img1 = Image.open(os.path.join(img_1_path,'img1/'+img1_frame+'.jpg'))
    img2 = Image.open(os.path.join(img_2_path,'img1/'+img2_frame+'.jpg'))
    if img1.mode != 'RGB':
        img1=img1.convert('RGB')
    if img2.mode != 'RGB':
        img2 = img2.convert('RGB')


    img_transform = standard_transforms.Compose([
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(*opt.MEAN_STD)
        ])

    img1 = img_transform(img1)
    img2 = img_transform(img2)
    img = [[img1,img2]]
    target = [target1[0],target2[0]]




    # In[4]:


    


        # ------------prepare enviroment------------


    os.environ["CUDA_VISIBLE_DEVICES"] = opt.GPU_ID
    torch.backends.cudnn.benchmark = True

    # ------------prepare data loader------------
    data_mode = opt.DATASET
    datasetting = import_module(f'datasets.setting.{data_mode}')
    cfg_data = datasetting.cfg_data

    # ------------Start Training------------
    cfg=opt
    print("model_path: ",cfg.MODEL_PATH)

    with torch.no_grad():
        net = DutyMOFANet(cfg, cfg_data)
        

        
        device = torch.device("cuda:"+str(torch.cuda.current_device()))
        

        state_dict = torch.load(cfg.MODEL_PATH,map_location=device)
        
        net.load_state_dict(state_dict["net"], strict=True)

        net.cuda()
        net.eval()

        generate_gt = GenerateGT(cfg)
        get_roi_and_matchinfo = get_ROI_and_MatchInfo( cfg.TRAIN_SIZE, cfg.ROI_RADIUS, feature_scale=cfg.FEATURE_SCALE)


    
        img,target = img[0],target
        # scene_name = target[0]['scene_name']
        img = torch.stack(img, 0).cuda()
        b, c, h, w = img.shape
        if h % 64 != 0:
            pad_h = 64 - h % 64
        else:
            pad_h = 0
        if w % 64 != 0:
            pad_w = 64 - w % 64
        else:
            pad_w = 0
        pad_dims = (0, pad_w, 0, pad_h)
        img = F.pad(img, pad_dims, "constant")
        img_pair_num = img.shape[0]//2

        

        den_scales, pred_map, mask, out_den, in_den, den_prob, io_prob, confidence, f_flow, b_flow, feature1, feature2, attn_1, attn_2 = net(img)

        pre_inflow, pre_outflow = \
            in_den.sum().detach().cpu(), out_den.sum().detach().cpu()
        
        target_ratio = pred_map.shape[2]/img.shape[2]

        for b in range(len(target)):
        
            
            for key,data in target[b].items():
                if torch.is_tensor(data):
                    target[b][key]=data.cuda()
        #    -----------gt generate metric computation------------------
            
        gt_den_scales = generate_gt.get_den(den_scales[0].shape, target, target_ratio, scale_num=len(den_scales))
        gt_den = gt_den_scales[0]
        
        assert pred_map.size() == gt_den.size()

        gt_io_map = torch.zeros(img_pair_num, 2, den_scales[0].size(2), den_scales[0].size(3)).cuda()

        


        gt_in_cnt = torch.zeros(img_pair_num).detach()
        gt_out_cnt = torch.zeros(img_pair_num).detach()

        assert pred_map.size() == gt_den.size()

        for pair_idx in range(img_pair_num):
            count_in_pair=[target[pair_idx * 2]['points'].size(0), target[pair_idx * 2+1]['points'].size(0)]
            
            if (np.array(count_in_pair) > 0).all() and (np.array(count_in_pair) < 4000).all():
                match_gt, _ = get_roi_and_matchinfo(target[pair_idx * 2], target[pair_idx * 2+1],'ab')

                gt_io_map, gt_in_cnt, gt_out_cnt \
                    = generate_gt.get_pair_io_map(pair_idx, target, match_gt, gt_io_map, gt_out_cnt, gt_in_cnt, target_ratio)
                    
                    # = generate_gt.get_pair_seg_map(pair_idx, target, match_gt, gt_io_map, gt_out_cnt, gt_in_cnt, target_ratio)
        gt_mask = (gt_io_map>0).float()
        restore_transform =standard_transforms.Compose([
            own_transforms.DeNormalize(*cfg.MEAN_STD),
            standard_transforms.ToPILImage()
        ])


        # save_results_mask(cfg, None, None, scene_name, (img1_frame, vi+cfg.TEST_INTERVALS), restore_transform, 0, 
        #                     img[0].clone().unsqueeze(0), img[1].clone().unsqueeze(0),\
        #                     pred_map[0].detach().cpu().numpy(), pred_map[1].detach().cpu().numpy(),out_den[0].detach().cpu().numpy(), in_den[0].detach().cpu().numpy(), gt_io_map[0].unsqueeze(0).detach().cpu().numpy(),\
        #                     (confidence[0,:,:,:]).unsqueeze(0).detach().cpu().numpy(),(confidence[1,:,:,:]).unsqueeze(0).detach().cpu().numpy(),\
        #                     f_flow , b_flow, [attn_1,attn_1,attn_1], [attn_2,attn_2,attn_2], den_scales, gt_den_scales, \
        #                     [mask,mask,mask], [gt_mask,gt_mask,gt_mask], [den_prob,den_prob,den_prob], [io_prob,io_prob,io_prob])
        




    # In[6]:


    gt_count_0 = len(target[0]['person_id'])
    gt_count_1 = len(target[1]['person_id'])

    count_0_scale = []
    count_1_scale = []

    for i in range(3):

        count_0_scale.append(round(torch.sum(den_scales[i][0]).item(),2))
        count_1_scale.append(round(torch.sum(den_scales[i][1]).item(),2))
    # count_0 = round(torch.sum(pred_map[0]).item(),2)
    # count_1 = round(torch.sum(pred_map[1]).item(),2)
    count_0 = sum(count_0_scale)
    count_1 = sum(count_1_scale)

    gt_in = len(match_gt['un_b'])
    gt_out = len(match_gt['un_a'])
    count_in = round(torch.sum(in_den).item(),2)
    count_out = round(torch.sum(out_den).item(),2)




 


    # In[7]:


    restore = restore_transform
    batch = 0 
    img0 = img[0].clone().unsqueeze(0)[0]
    img1 = img[1].clone().unsqueeze(0)[0]
    den0 =  pred_map[0].detach().cpu().numpy()[0]
    den1 = pred_map[1].detach().cpu().numpy()[0]
    out_map = out_den[0].detach().cpu().numpy()[0]
    # in_map = in_den[0].detach().cpu().numpy()[0]
    # gt_io_map = gt_io_map[0].unsqueeze(0).detach().cpu().numpy()[0]
    conf0=(confidence[0,:,:,:]).unsqueeze(0).detach().cpu().numpy()[0]
    conf1=(confidence[1,:,:,:]).unsqueeze(0).detach().cpu().numpy()[0]
    gt_den_scales=gt_den_scales
    mask=[mask,mask,mask]
    gt_mask=[gt_mask,gt_mask,gt_mask]
    den_probs=[den_prob,den_prob,den_prob]
    io_probs= [io_prob,io_prob,io_prob]



    # In[8]:


    gaussian_kernel = 31
    gaussian_sigma = 10


    cfg.TRAIN_BATCH_SIZE = 1

    pil_to_tensor = standard_transforms.ToTensor()

    UNIT_H , UNIT_W = img0.size(1), img0.size(2)
    # for idx, tensor in enumerate(zip(img0.cpu().data, img1.cpu().data,pred_map0, gt_map0, pred_map1, gt_map1, \
    #                                  pred_mask_out, gt_mask_out, pred_mask_in, gt_mask_in, attn_1, attn_2)):

    if cfg.MODE == 'test':
        cfg.TRAIN_BATCH_SIZE = cfg.VAL_BATCH_SIZE

    COLOR_MAP = [
        [255, 0, 0],
        [0, 0, 255],
        [0, 255, 255],
    ]
    COLOR_MAP = np.array(COLOR_MAP, dtype="uint8")
    COLOR_MAP_CONF = [
        [255, 255, 0],
        [255, 0, 255],
        [0, 255, 255],
    ]
    COLOR_MAP_CONF = np.array(COLOR_MAP_CONF, dtype="uint8")



    den_scales_1_map = []
    gt_den_scales_1_map = []
    den_scales_2_map = []
    gt_den_scales_2_map = []
    mask_in_scales_1_map = []
    gt_mask_in_scales_1_map = []
    mask_out_scales_1_map = []
    gt_mask_out_scales_1_map = []
    # den_prob_map_1 = []
    # io_prob_map_1 = []
    # den_prob_map_2 = []
    # io_prob_map_2 = []




    pil_input0 = restore(img0.cpu().data)
    pil_input1 = restore(img1.cpu().data)


    a = [0,0,0]


    for i in range(len(den_scales)):

        den_scale_1 = den_scales[i][0].detach().cpu().numpy()[0]
        den_scale_2 = den_scales[i][1].detach().cpu().numpy()[0]
        gt_den_scale_1 = gt_den_scales[i][0].detach().cpu().numpy()[0]
        gt_den_scale_2 = gt_den_scales[i][1].detach().cpu().numpy()[0]
        den_scale_1 = cv2.GaussianBlur(den_scale_1, (int(gaussian_kernel/2**i+a[i]),int(gaussian_kernel/2**i+a[i]),),int(10/2**i))
        den_scale_2 = cv2.GaussianBlur(den_scale_2, (int(gaussian_kernel/2**i+a[i]),int(gaussian_kernel/2**i+a[i]),),int(10/2**i))
        gt_den_scale_1 = cv2.GaussianBlur(gt_den_scale_1, (int(gaussian_kernel/2**i+a[i]),int(gaussian_kernel/2**i+a[i]),),int(10/2**i))
        gt_den_scale_2 = cv2.GaussianBlur(gt_den_scale_2, (int(gaussian_kernel/2**i+a[i]),int(gaussian_kernel/2**i+a[i]),),int(10/2**i))



        den_scale_1 = cv2.resize(cv2.applyColorMap((255 * den_scale_1 / (den_scale_1.max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET), (UNIT_W, UNIT_H)) 
        den_scale_2 = cv2.resize(cv2.applyColorMap((255 * den_scale_2 / (den_scale_2.max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET), (UNIT_W, UNIT_H)) 
        gt_den_scale_1 = cv2.resize(cv2.applyColorMap((255 * gt_den_scale_1 / (gt_den_scale_1.max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET), (UNIT_W, UNIT_H)) 
        gt_den_scale_2 = cv2.resize(cv2.applyColorMap((255 * gt_den_scale_2 / (gt_den_scale_2.max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET), (UNIT_W, UNIT_H)) 
        den_scale_1 = cv2.cvtColor(den_scale_1, cv2.COLOR_BGR2RGB)
        den_scale_2 = cv2.cvtColor(den_scale_2, cv2.COLOR_BGR2RGB)
        gt_den_scale_1 = cv2.cvtColor(gt_den_scale_1, cv2.COLOR_BGR2RGB)
        gt_den_scale_2 = cv2.cvtColor(gt_den_scale_2, cv2.COLOR_BGR2RGB)

        den_scales_1_map.append(den_scale_1)
        den_scales_2_map.append(den_scale_2)
        gt_den_scales_1_map.append(gt_den_scale_1)
        gt_den_scales_2_map.append(gt_den_scale_2)


        ########## mask ###############
        mask_out_scale_1 = mask[i][0,:,:,:].detach().cpu().numpy()[0]
        mask_in_scale_1 =  mask[i][cfg.TRAIN_BATCH_SIZE,:,:,:].detach().cpu().numpy()[0]
        
        mask_out_scale_1 = cv2.GaussianBlur(mask_out_scale_1, (gaussian_kernel,gaussian_kernel,),gaussian_sigma)
        mask_in_scale_1 = cv2.GaussianBlur(mask_in_scale_1, (gaussian_kernel,gaussian_kernel,),gaussian_sigma)
        # mask_out_scale_1[mask_out_scale_1>0.2] = 1
        # mask_in_scale_1[mask_in_scale_1>0.2] = 1

        
        gt_mask_out_scale_1 = gt_mask[i][0,0:1,:,:].detach().cpu().numpy()[0]
        gt_mask_in_scale_1 = gt_mask[i][0,1:2,:,:].detach().cpu().numpy()[0]
        
        gt_mask_out_scale_1 = cv2.GaussianBlur(gt_mask_out_scale_1, (gaussian_kernel,gaussian_kernel,),gaussian_sigma)
        gt_mask_in_scale_1 = cv2.GaussianBlur(gt_mask_in_scale_1, (gaussian_kernel,gaussian_kernel,),gaussian_sigma)
        gt_mask_out_scale_1[gt_mask_out_scale_1>0.15] = 1
        gt_mask_in_scale_1[gt_mask_in_scale_1>0.15] = 1

        
        # mask_out_scale_1 = np.clip((cv2.add(10*mask_out_scale_1,gaussian_sigma)), 0 , 255)
        # mask_in_scale_1 = np.clip((cv2.add(10*mask_in_scale_1,gaussian_sigma)), 0 , 255)
        # mask_out_scale_1 = (mask_out_scale_1-mask_out_scale_1.min())/(mask_out_scale_1.max()-mask_out_scale_1.min())
        # mask_in_scale_1 = (mask_in_scale_1-mask_in_scale_1.min())/(mask_in_scale_1.max()-mask_in_scale_1.min())


        mask_out_scale_1 = cv2.resize(cv2.applyColorMap((255 * mask_out_scale_1 / (mask_out_scale_1.max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_HOT), (UNIT_W, UNIT_H)) 
        mask_in_scale_1 = cv2.resize(cv2.applyColorMap((255 * mask_in_scale_1 / (mask_in_scale_1.max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_HOT), (UNIT_W, UNIT_H)) 
        gt_mask_out_scale_1 = cv2.resize(cv2.applyColorMap((255 * gt_mask_out_scale_1 / (gt_mask_out_scale_1.max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_HOT), (UNIT_W, UNIT_H)) 
        gt_mask_in_scale_1 = cv2.resize(cv2.applyColorMap((255 * gt_mask_in_scale_1 / (gt_mask_in_scale_1.max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_HOT), (UNIT_W, UNIT_H)) 
        
        
        

        mask_out_scale_1 = cv2.cvtColor(mask_out_scale_1, cv2.COLOR_BGR2RGB)
        mask_in_scale_1 = cv2.cvtColor(mask_in_scale_1, cv2.COLOR_BGR2RGB)
        gt_mask_out_scale_1 = cv2.cvtColor(gt_mask_out_scale_1, cv2.COLOR_BGR2RGB)
        gt_mask_in_scale_1 = cv2.cvtColor(gt_mask_in_scale_1, cv2.COLOR_BGR2RGB)
        

        mask_out_scales_1_map.append(mask_out_scale_1)
        mask_in_scales_1_map.append(mask_in_scale_1)
        gt_mask_out_scales_1_map.append(gt_mask_out_scale_1)
        gt_mask_in_scales_1_map.append(gt_mask_in_scale_1)




        den_prob_1 = den_probs[i][0,:,:,:].detach().cpu().numpy()
        den_prob_2 =  den_probs[i][cfg.TRAIN_BATCH_SIZE,:,:,:].detach().cpu().numpy()

        io_prob_1 = io_probs[i][0,:,:,:].detach().cpu().numpy()
        io_prob_2 =  io_probs[i][cfg.TRAIN_BATCH_SIZE,:,:,:].detach().cpu().numpy()

        conf0[i] = cv2.GaussianBlur(conf0[i], (gaussian_kernel,gaussian_kernel,),gaussian_sigma)
        conf1[i] = cv2.GaussianBlur(conf1[i], (gaussian_kernel,gaussian_kernel,),gaussian_sigma)

        


    # ratio = UNIT_H/den0.shape[0]
    den0_map = cv2.GaussianBlur(den0, (gaussian_kernel,gaussian_kernel,),gaussian_sigma)
    den0_map = cv2.resize(cv2.applyColorMap((255 * den0_map / (den0_map.max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET), (UNIT_W, UNIT_H)) 
    den1_map = cv2.GaussianBlur(den1, (gaussian_kernel,gaussian_kernel,),gaussian_sigma)
    den1_map = cv2.resize(cv2.applyColorMap((255 * den1_map / (den1_map.max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET), (UNIT_W, UNIT_H)) 

    # out_map = cv2.resize(cv2.applyColorMap((255 * out_map / (out_map.max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET), (UNIT_W, UNIT_H)) 
    # in_map = cv2.resize(cv2.applyColorMap((255 * in_map / (in_map.max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET), (UNIT_W, UNIT_H)) 

    # gt_out_map = cv2.resize(cv2.applyColorMap((255 * gt_io_map[0] / (gt_io_map[0].max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET), (UNIT_W, UNIT_H)) 
    # gt_in_map = cv2.resize(cv2.applyColorMap((255 * gt_io_map[1] / (gt_io_map[1].max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET), (UNIT_W, UNIT_H)) 


    def drawPoint(img, points):
        points = np.ceil(points.cpu().numpy())
        point_size = 1
        point_color = (0,255,0)
        thickness = 26
        for point in points:
            cv2.circle(img, [int(point[0]),int(point[1])], point_size, point_color, thickness)
        return img



    dot_map0 = np.full((conf0.shape[1],conf0.shape[2],3), 255).astype(np.uint8)
    dot_map0 = drawPoint(dot_map0, target[0]["points"])
    conf_map0 = np.argmax(conf0, axis=0)
    conf_map0 = cv2.resize(COLOR_MAP_CONF[conf_map0].squeeze(),  (UNIT_W, UNIT_H))
    conf_map0_dot = 255 - conf_map0 * np.repeat(((dot_map0[:,:,0])<255).squeeze(),3,axis=1).reshape(UNIT_H, UNIT_W, 3)


    dot_map1 = np.full((conf1.shape[1],conf1.shape[2],3), 255).astype(np.uint8)
    dot_map1 = drawPoint(dot_map0, target[1]["points"])
    conf_map1 = np.argmax(conf1, axis=0)
    conf_map1 = cv2.resize(COLOR_MAP_CONF[conf_map1].squeeze(),  (UNIT_W, UNIT_H))
    conf_map1_dot = 255 - conf_map1 * np.repeat((((dot_map0[:,:,0])<255)).squeeze(),3,axis=1).reshape(UNIT_H, UNIT_W, 3)

    conf_0_scale_0 = cv2.resize(cv2.applyColorMap((255 *  conf0[0] / ( conf0[0].max() + 1e-10)).astype(np.uint8), cv2.COLORMAP_JET), (UNIT_W, UNIT_H))
    conf_0_scale_1 = cv2.resize(cv2.applyColorMap((255 *  conf0[1] / ( conf0[1].max() + 1e-10)).astype(np.uint8), cv2.COLORMAP_JET), (UNIT_W, UNIT_H))
    conf_0_scale_2 = cv2.resize(cv2.applyColorMap((255 *  conf0[2] / ( conf0[2].max() + 1e-10)).astype(np.uint8), cv2.COLORMAP_JET), (UNIT_W, UNIT_H))
    conf_1_scale_0 = cv2.resize(cv2.applyColorMap((255 *  conf1[0] / ( conf0[0].max() + 1e-10)).astype(np.uint8), cv2.COLORMAP_JET), (UNIT_W, UNIT_H))
    conf_1_scale_1 = cv2.resize(cv2.applyColorMap((255 *  conf1[1] / ( conf0[1].max() + 1e-10)).astype(np.uint8), cv2.COLORMAP_JET), (UNIT_W, UNIT_H))
    conf_1_scale_2 = cv2.resize(cv2.applyColorMap((255 *  conf1[2] / ( conf0[2].max() + 1e-10)).astype(np.uint8), cv2.COLORMAP_JET), (UNIT_W, UNIT_H))


    def drawText(array, text='', w=0.6):
        return cv2.putText(array, text, (int(UNIT_W*w), int(UNIT_H*0.9)),cv2.FONT_HERSHEY_SIMPLEX,4,(255,255,255),15)


    pil_input0 = np.array(pil_input0)
    pil_input1 = np.array(pil_input1)
    pil_input0 = drawPoint(pil_input0, target[0]["points"])
    pil_input1 = drawPoint(pil_input1, target[1]["points"])

    pil_input0= drawText( pil_input0, 'GT: {:.2f}'.format(gt_count_0))
    pil_input1= drawText( pil_input1, 'GT: {:.2f}'.format(gt_count_1))





    for i in range(len(den_scales_1_map)):
        den_scales_1_map[i]= drawText( den_scales_1_map[i], 'Pre: {:.2f}'.format(count_0_scale[i]))
        gt_den_scales_1_map[i]= drawText( gt_den_scales_1_map[i], 'GT: {:.2f}'.format(gt_count_0))
        den_scales_2_map[i]= drawText( den_scales_2_map[i], 'Pre: {:.2f}'.format(count_1_scale[i]))
        gt_den_scales_2_map[i]= drawText( gt_den_scales_2_map[i], 'GT: {:.2f}'.format(gt_count_1))


        den_scales_1_map[i]= Image.fromarray(den_scales_1_map[i])
        gt_den_scales_1_map[i]= Image.fromarray(gt_den_scales_1_map[i])
        den_scales_2_map[i]= Image.fromarray(den_scales_2_map[i])
        gt_den_scales_2_map[i]= Image.fromarray(gt_den_scales_2_map[i])
        
        gt_mask_out_scales_1_map[i]= drawText( gt_mask_out_scales_1_map[i], 'GT Out: {:.2f}'.format(gt_out),0.53)
        gt_mask_in_scales_1_map[i]= drawText( gt_mask_in_scales_1_map[i], 'GT In: {:.2f}'.format(gt_in),0.58)
        mask_out_scales_1_map[i]= drawText( mask_out_scales_1_map[i], 'Pre Out: {:.2f}'.format(count_out),0.5)
        mask_in_scales_1_map[i]= drawText( mask_in_scales_1_map[i], 'Pre In: {:.2f}'.format(count_in),0.55)

        mask_out_scales_1_map[i]= Image.fromarray(mask_out_scales_1_map[i])
        mask_in_scales_1_map[i]= Image.fromarray(mask_in_scales_1_map[i])
        gt_mask_out_scales_1_map[i]= Image.fromarray(gt_mask_out_scales_1_map[i])
        gt_mask_in_scales_1_map[i]= Image.fromarray(gt_mask_in_scales_1_map[i])




    pil_input0 = Image.fromarray(pil_input0)
    pil_input1 = Image.fromarray(pil_input1)

    den0_map= drawText( den0_map, 'Pre: {:.2f}'.format(count_0))
    den1_map= drawText( den1_map, 'Pre: {:.2f}'.format(count_1))


    den0_map = Image.fromarray(cv2.cvtColor(den0_map, cv2.COLOR_BGR2RGB))
    den1_map = Image.fromarray(cv2.cvtColor(den1_map, cv2.COLOR_BGR2RGB))
    # out_map = Image.fromarray(cv2.cvtColor(out_map, cv2.COLOR_BGR2RGB))
    # in_map = Image.fromarray(cv2.cvtColor(in_map, cv2.COLOR_BGR2RGB))
    # gt_out_map = Image.fromarray(cv2.cvtColor(gt_out_map, cv2.COLOR_BGR2RGB))
    # gt_in_map = Image.fromarray(cv2.cvtColor(gt_in_map, cv2.COLOR_BGR2RGB))
    conf_map0 = Image.fromarray(cv2.cvtColor(conf_map0, cv2.COLOR_BGR2RGB))
    conf_map1 = Image.fromarray(cv2.cvtColor(conf_map1, cv2.COLOR_BGR2RGB))
    conf_map0_dot = Image.fromarray(cv2.cvtColor(conf_map0_dot, cv2.COLOR_BGR2RGB))
    conf_map1_dot = Image.fromarray(cv2.cvtColor(conf_map1_dot, cv2.COLOR_BGR2RGB))

    conf_0_scale_0 = Image.fromarray(cv2.cvtColor(conf_0_scale_0, cv2.COLOR_BGR2RGB))
    conf_0_scale_1 = Image.fromarray(cv2.cvtColor(conf_0_scale_1, cv2.COLOR_BGR2RGB))
    conf_0_scale_2 = Image.fromarray(cv2.cvtColor(conf_0_scale_2, cv2.COLOR_BGR2RGB))

    conf_1_scale_0 = Image.fromarray(cv2.cvtColor(conf_1_scale_0, cv2.COLOR_BGR2RGB))
    conf_1_scale_1 = Image.fromarray(cv2.cvtColor(conf_1_scale_1, cv2.COLOR_BGR2RGB))
    conf_1_scale_2 = Image.fromarray(cv2.cvtColor(conf_1_scale_2, cv2.COLOR_BGR2RGB))






    # In[9]:


    folder = f"../visualization/CARLA/{scene_name}"
    os.makedirs(folder,exist_ok=True)
    os.makedirs(os.path.join(folder,'img1_denscale'),exist_ok=True)
    # os.makedirs(os.path.join(folder,'img1_GTdenscale'),exist_ok=True)
    os.makedirs(os.path.join(folder,'img2_denscale'),exist_ok=True)
    # os.makedirs(os.path.join(folder,'img2_GTdenscale'),exist_ok=True)
    os.makedirs(os.path.join(folder,'img1_confscale'),exist_ok=True)
    os.makedirs(os.path.join(folder,'img1_confscale'),exist_ok=True)
    os.makedirs(os.path.join(folder,'img1_confscale'),exist_ok=True)
    os.makedirs(os.path.join(folder,'img2_confscale'),exist_ok=True)
    os.makedirs(os.path.join(folder,'img2_confscale'),exist_ok=True)
    os.makedirs(os.path.join(folder,'img2_confscale'),exist_ok=True)


    # for i in range(len(den_scales_1_map)):
    #     den_scales_1_map[i].save(os.path.join(folder,f"img1_denscale/{i}.jpg"))
    #     den_scales_2_map[i].save(os.path.join(folder,f"img2_denscale/{i}.jpg"),None)
    mask_out_scales_1_map[0].save(os.path.join(folder,f"{img1_frame}_{img2_frame}_img1_mask_out.jpg"),None)
    mask_in_scales_1_map[0].save(os.path.join(folder,f"{img1_frame}_{img2_frame}_img2_mask_in.jpg"),None)
    gt_mask_out_scales_1_map[0].save(os.path.join(folder,f"{img1_frame}_{img2_frame}_img1_mask_out_gt.jpg"),None)
    gt_mask_in_scales_1_map[0].save(os.path.join(folder,f"{img1_frame}_{img2_frame}_img2_mask_in_gt.jpg"),None)

    # gt_den_scales_1_map[0].save(os.path.join(folder,f"img1_den_gt.jpg"),None)
    # gt_den_scales_2_map[0].save(os.path.join(folder,f"img2_den_gt.jpg"),None)


    # pil_input0.save(os.path.join(folder,f"img1.jpg"),None)
    # pil_input1.save(os.path.join(folder,f"img2.jpg"),None)
    # den0_map.save(os.path.join(folder,f"img1_den.jpg"),None)
    # den1_map.save(os.path.join(folder,f"img2_den.jpg"),None)
    # # out_map.save(os.path.join(folder,f"img1_out_den.jpg"),None)
    # # in_map.save(os.path.join(folder,f"img2_in_den.jpg"),None)
    # # gt_out_map.save(os.path.join(folder,f"img1_out_den_gt.jpg"),None)
    # # gt_in_map.save(os.path.join(folder,f"img2_in_den_gt.jpg"),None)
    # conf_map0_dot.save(os.path.join(folder,f"img1_confdot.jpg"),None)
    # conf_map1_dot.save(os.path.join(folder,f"img2_confdot.jpg"),None)
    # conf_0_scale_0.save(os.path.join(folder,f"img1_confscale/0.jpg"),None)
    # conf_0_scale_1.save(os.path.join(folder,f"img1_confscale/1.jpg"),None)
    # conf_0_scale_2.save(os.path.join(folder,f"img1_confscale/2.jpg"),None)
    # conf_1_scale_0.save(os.path.join(folder,f"img2_confscale/0.jpg"),None)
    # conf_1_scale_1.save(os.path.join(folder,f"img2_confscale/1.jpg"),None)
    # conf_1_scale_2.save(os.path.join(folder,f"img2_confscale/2.jpg"),None)
    index +=1
    # except:
    #     break


