import os
import math
import numpy as np
import time
import random
import shutil
import cv2
from PIL import Image
import pdb
import torch
from torch import nn
import torchvision.utils as vutils
import torchvision.transforms as standard_transforms
from .flow_viz import *
from tensorboardX import SummaryWriter



def adjust_learning_rate(optimizer, epoch,base_lr1=0, base_lr2=0, power=0.9):
    lr1 =  base_lr1 * power ** ((epoch-1))
    lr2 =  base_lr2 * power ** ((epoch - 1))
    optimizer.param_groups[0]['lr'] = lr1
    optimizer.param_groups[1]['lr'] = lr2
    optimizer.param_groups[2]['lr'] = lr2

    return lr1 , lr2



def weights_normal_init(*models):
    for model in models:
        dev=0.01
        if isinstance(model, list):
            for m in model:
                weights_normal_init(m, dev)
        else:
            for m in model.modules():            
                if isinstance(m, nn.Conv2d):        
                    m.weight.data.normal_(0.0, dev)
                    if m.bias is not None:
                        m.bias.data.fill_(0.0)
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0.0, dev)


#used
def logger(cfg, exp_path, exp_name, work_dir, exception, resume=False):

    # exp_path = cfg.EXP_PATH
    # exp_name = cfg.EXP_NAME

    
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    writer = SummaryWriter(exp_path+ '/' + exp_name)
    log_file = exp_path + '/' + exp_name + '/' + exp_name + '.txt'
    
    # cfg_file = open('./config.py',"r")  
    # cfg_lines = cfg_file.readlines()
    
    with open(log_file, 'a') as f:
        for k in list(vars(cfg).keys()):
            f.write(f'{k}: {vars(cfg)[k]}' + '\n')
        f.write('\n\n')

    if not resume:
        copy_cur_env(work_dir, exp_path+ '/' + exp_name +'/src', exception)

    return writer, log_file


def logger_txt(log_file,epoch,iter,scores):
    snapshot_name = 'ep_%d_iter_%d' % (epoch,iter)
    for key, data in scores.items():
        snapshot_name+= ('_'+ key+'_%3f'%data)
    with open(log_file, 'a') as f:
        f.write('='*15 + '+'*15 + '='*15 + '\n\n')
        f.write(snapshot_name + '\n')
        f.write('[')
        for key, data in scores.items():
            f.write(' '+ key+' %.2f'% data)
        f.write('\n')
        f.write('='*15 + '+'*15 + '='*15 + '\n\n')

def save_test_logger(cfg, exp_path,cnt_result, final_result):
    
    # dir = os.path.join(exp_path,cfg.task,cfg.model_path.split('/')[-2],os.path.basename(cfg.model_path).split('_')[3])
    dir = cfg.model_path.replace('exp', exp_path).replace(cfg.model_path.split('/')[-1],'' )

    if not os.path.isdir(dir):
        os.makedirs(dir)
   

    log_file = os.path.join(dir, 'log.txt')

    with open(log_file, 'a') as f:
        f.write(f'ep: {os.path.basename(cfg.model_path).split("_")[1]}    iter: {os.path.basename(cfg.model_path).split("_")[3]}\n\n')
        f.write(f'model_path: {cfg.model_path}  \ntest_interval: {cfg.test_intervals}\n\n')
        f.write(f'Prev vs. GT: {cnt_result}\n\n')
        for k in final_result.keys():
            f.write(f'{k}: {final_result[k]}\n\n')





# def save_results_mask(cfg, exp_path, exp_name, scene_name, iter, restore, batch, img0, img1, pred_map0, gt_map0,\
#                         pred_map1, gt_map1, pred_mask_out, gt_mask_out, pred_mask_in, gt_mask_in, f_flow,b_flow, attn_1,attn_2,den_scales, gt_den_scales):
                        #attn_ref, attn_sou): 
def save_results_mask(cfg, exp_path, exp_name, scene_name, iter, restore, batch, img0, img1, den0, den1, out_map, in_map, conf0, conf1,\
                       f_flow,b_flow, attn_1,attn_2,den_scales, gt_den_scales, mask, gt_mask, den_probs, io_probs):

    pil_to_tensor = standard_transforms.ToTensor()

    UNIT_H , UNIT_W = img0.size(2), img0.size(3)
    # for idx, tensor in enumerate(zip(img0.cpu().data, img1.cpu().data,pred_map0, gt_map0, pred_map1, gt_map1, \
    #                                  pred_mask_out, gt_mask_out, pred_mask_in, gt_mask_in, attn_1, attn_2)):

    COLOR_MAP = [
        [255, 0, 0],
        [0, 0, 255],
        [0, 255, 255],
    ]
    COLOR_MAP = np.array(COLOR_MAP, dtype="uint8")
    COLOR_MAP_CONF = [
        [0, 255, 0],
        [230, 216, 173], #sky blue
        [203, 192, 255], #pink
    ]
    COLOR_MAP_CONF = np.array(COLOR_MAP_CONF, dtype="uint8")

    for idx, tensor in enumerate(zip(img0.cpu().data, img1.cpu().data, den0, den1, out_map, in_map, conf0, conf1)):
        if idx > 1:  # show only one group
            break

        f_flow_map = []
        b_flow_map = []
        # attn_1_map = []
        # attn_2_map = []
        feature_1_map = []
        align_feature_2_map = []
        feature_2_map = []
        align_feature_1_map = []
        den_scales_1_map = []
        gt_den_scales_1_map = []
        den_scales_2_map = []
        gt_den_scales_2_map = []
        mask_in_scales_1_map = []
        gt_mask_in_scales_1_map = []
        mask_out_scales_1_map = []
        gt_mask_out_scales_1_map = []
        den_prob_map_1 = []
        io_prob_map_1 = []
        den_prob_map_2 = []
        io_prob_map_2 = []




        pil_input0 = restore(tensor[0])
        pil_input1 = restore(tensor[1])

        for i in range(len(f_flow)):
            f = f_flow[i][batch].permute(1,2,0).detach().cpu().numpy()
            b = b_flow[i][batch].permute(1,2,0).detach().cpu().numpy()
            f_flow_map.append(flow_to_image(f))
            b_flow_map.append(flow_to_image(b))
            # print(attn_1[i][0][0].shape)
        for i in range(len(attn_1)):

            a1 = np.max(attn_1[i][0][batch].detach().cpu().numpy(), axis=0)
            a2 = np.max(attn_2[i][0][batch].detach().cpu().numpy(), axis=0)
            a1 = cv2.resize(a1[None,:].transpose(1,2,0), (UNIT_W, UNIT_H))
            a2 = cv2.resize(a2[None,:].transpose(1,2,0), (UNIT_W, UNIT_H))
            a1 = cv2.applyColorMap((255 * a1 / (a1.max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
            a2 = cv2.applyColorMap((255 * a2 / (a2.max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)

            a3 = np.max(attn_1[i][1][batch].detach().cpu().numpy(), axis=0)
            a4 = np.max(attn_2[i][1][batch].detach().cpu().numpy(), axis=0)
            a3 = cv2.resize(a3[None,:].transpose(1,2,0), (UNIT_W, UNIT_H))
            a4 = cv2.resize(a4[None,:].transpose(1,2,0), (UNIT_W, UNIT_H))
            a3 = cv2.applyColorMap((255 * a3 / (a3.max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
            a4 = cv2.applyColorMap((255 * a4 / (a4.max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)

            a1 = cv2.cvtColor(a1, cv2.COLOR_BGR2GRAY)
            a2=cv2.cvtColor(a2, cv2.COLOR_BGR2GRAY)
            a3=cv2.cvtColor(a3, cv2.COLOR_BGR2GRAY)
            a4=cv2.cvtColor(a4, cv2.COLOR_BGR2GRAY)
            
            
            # attn_1_map.append(a1)
            # attn_2_map.append(a2)
            feature_1_map.append(a1)
            align_feature_2_map.append(a3)
            feature_2_map.append(a2)
            align_feature_1_map.append(a4)


        for i in range(len(den_scales)):
            den_scale_1 = den_scales[i][0].detach().cpu().numpy()
            den_scale_2 = den_scales[i][1].detach().cpu().numpy()
            gt_den_scale_1 = gt_den_scales[i][0].detach().cpu().numpy()
            gt_den_scale_2 = gt_den_scales[i][1].detach().cpu().numpy()
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
            # mask_out_scale_1 = mask[i][0,:,:,:].detach().cpu().numpy()
            # mask_in_scale_1 =  mask[i][2,:,:,:].detach().cpu().numpy()
            

            mask_out_scale_1 = np.argmax(mask[i][0,:,:,:].detach().cpu().numpy(), axis=0)
            mask_in_scale_1 =  np.argmax(mask[i][cfg.TRAIN_BATCH_SIZE,:,:,:].detach().cpu().numpy(),axis=0)
            
            gt_mask_out_scale_1 = gt_mask[i][0,0:1,:,:].detach().cpu().numpy()
            gt_mask_in_scale_1 = gt_mask[i][0,1:2,:,:].detach().cpu().numpy()

            mask_out_scale_1 = cv2.resize(COLOR_MAP[mask_out_scale_1].squeeze(),  (UNIT_W, UNIT_H))
            mask_in_scale_1 = cv2.resize(COLOR_MAP[mask_in_scale_1].squeeze(),  (UNIT_W, UNIT_H))
            gt_mask_out_scale_1 = cv2.resize(COLOR_MAP[gt_mask_out_scale_1].squeeze(),  (UNIT_W, UNIT_H))
            gt_mask_in_scale_1 = cv2.resize(COLOR_MAP[gt_mask_in_scale_1].squeeze(),  (UNIT_W, UNIT_H))




            # mask_out_scale_1 = cv2.resize(cv2.applyColorMap((255 * mask_out_scale_1 / (mask_out_scale_1.max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET), (UNIT_W, UNIT_H)) 
            # mask_in_scale_1 = cv2.resize(cv2.applyColorMap((255 * mask_in_scale_1 / (mask_in_scale_1.max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET), (UNIT_W, UNIT_H)) 
            # gt_mask_out_scale_1 = cv2.resize(cv2.applyColorMap((255 * gt_mask_out_scale_1 / (gt_mask_out_scale_1.max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET), (UNIT_W, UNIT_H)) 
            # gt_mask_in_scale_1 = cv2.resize(cv2.applyColorMap((255 * gt_mask_in_scale_1 / (gt_mask_in_scale_1.max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET), (UNIT_W, UNIT_H)) 
            
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
            
            den_prob_1 = cv2.resize(cv2.applyColorMap((255 * den_prob_1 / (den_prob_1.max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET), (UNIT_W, UNIT_H)) 
            den_prob_2 = cv2.resize(cv2.applyColorMap((255 * den_prob_2 / (den_prob_2.max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET), (UNIT_W, UNIT_H)) 
            io_prob_1 = cv2.resize(cv2.applyColorMap((255 * io_prob_1 / (io_prob_1.max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET), (UNIT_W, UNIT_H)) 
            io_prob_2 = cv2.resize(cv2.applyColorMap((255 * io_prob_2 / (io_prob_2.max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET), (UNIT_W, UNIT_H)) 
            den_prob_1 = cv2.cvtColor(den_prob_1, cv2.COLOR_BGR2RGB)
            den_prob_2 = cv2.cvtColor(den_prob_2, cv2.COLOR_BGR2RGB)
            io_prob_1 = cv2.cvtColor(io_prob_1, cv2.COLOR_BGR2RGB)
            io_prob_2 = cv2.cvtColor(io_prob_2, cv2.COLOR_BGR2RGB)

            den_prob_map_1.append(den_prob_1)
            den_prob_map_2.append(den_prob_2)
            io_prob_map_1.append(io_prob_1)
            io_prob_map_2.append(io_prob_2)



       
        # ratio = UNIT_H/tensor[2].shape[0]
        den0_map = cv2.resize(cv2.applyColorMap((255 * tensor[2] / (tensor[2].max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET), (UNIT_W, UNIT_H)) 
        den1_map = cv2.resize(cv2.applyColorMap((255 * tensor[3] / (tensor[3].max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET), (UNIT_W, UNIT_H)) 

        out_map = cv2.resize(cv2.applyColorMap((255 * tensor[4] / (tensor[4].max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET), (UNIT_W, UNIT_H)) 
        in_map = cv2.resize(cv2.applyColorMap((255 * tensor[5] / (tensor[5].max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET), (UNIT_W, UNIT_H)) 
        
        conf_map0 = np.argmax(tensor[6], axis=0)
        conf_map0 = cv2.resize(COLOR_MAP_CONF[conf_map0].squeeze(),  (UNIT_W, UNIT_H))
        mask0 = gt_mask[0][0,0:1,:,:].detach().cpu().numpy()
        conf_map0 = conf_map0 * np.repeat(((mask0==1) | (mask0==2)).squeeze(),3,axis=1).reshape(UNIT_H, UNIT_W, 3)
        # gt_conf_map0 = np.argmax(tensor[7], axis=0)
        # gt_conf_map0 = cv2.resize(COLOR_MAP_CONF[gt_conf_map0].squeeze(),  (UNIT_W, UNIT_H))
        conf_map1 = np.argmax(tensor[7], axis=0)
        conf_map1 = cv2.resize(COLOR_MAP_CONF[conf_map1].squeeze(),  (UNIT_W, UNIT_H))
        mask1 = gt_mask[0][0,1:2,:,:].detach().cpu().numpy()
        conf_map1 = conf_map1 * np.repeat(((mask1==1) | (mask1==2)).squeeze(),3,axis=1).reshape(UNIT_H, UNIT_W, 3)

        # gt_conf_map1 = np.argmax(tensor[9], axis=0)
        # gt_conf_map1 = cv2.resize(COLOR_MAP_CONF[gt_conf_map1].squeeze(),  (UNIT_W, UNIT_H))


        # pred_color_map0 = cv2.resize(cv2.applyColorMap((255 * tensor[2] / (tensor[2].max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET), (UNIT_W, UNIT_H)) 
        # gt_color_map0 = cv2.resize(cv2.applyColorMap((255 * tensor[3] / (tensor[3].max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET), (UNIT_W, UNIT_H)) 
        # pred_color_map1 = cv2.resize(cv2.applyColorMap((255 * tensor[4] / (tensor[4].max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET), (UNIT_W, UNIT_H))
        # gt_color_map1 = cv2.resize(cv2.applyColorMap((255 * tensor[5] / (tensor[5].max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET), (UNIT_W, UNIT_H)) 
        # pred_out_color = cv2.resize(cv2.applyColorMap((255 * tensor[6] / (tensor[6].max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET), (UNIT_W, UNIT_H))
        # gt_out_color = cv2.resize(cv2.applyColorMap((255 * tensor[7]/ (tensor[7].max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET), (UNIT_W, UNIT_H))
        # pred_in_color = cv2.resize(cv2.applyColorMap((255 * tensor[8] / (tensor[8].max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET), (UNIT_W, UNIT_H))
        # gt_in_color = cv2.resize(cv2.applyColorMap((255 * tensor[9]/ (tensor[9].max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET), (UNIT_W, UNIT_H))

   
        
        pil_input0 = np.array(pil_input0)
        pil_input1 = np.array(pil_input1)
        # pil_input0 = cv2.cvtColor(pil_input0, cv2.COLOR_LAB2RGB)
        # pil_input1 = cv2.cvtColor(pil_input1, cv2.COLOR_LAB2RGB)


        # for i in range(3):
        for i in range(len(f_flow_map)):


            # f_flow_map[i] = cv2.addWeighted(pil_input0, 0.5, f_flow_map[i], 0.5, 0)
            f_flow_map[i] = Image.fromarray(f_flow_map[i])
            # b_flow_map[i] = cv2.addWeighted(pil_input1, 0.5, b_flow_map[i], 0.5, 0)
            b_flow_map[i] = Image.fromarray(b_flow_map[i])
            # attn_1_map[i] = cv2.addWeighted(pil_input0, 0.5, attn_1_map[i], 0.5, 0)
            # attn_1_map[i] = Image.fromarray(attn_1_map[i])
            # attn_2_map[i] = cv2.addWeighted(pil_input1, 0.5,attn_2_map[i], 0.5, 0)
            # attn_2_map[i] = Image.fromarray(attn_2_map[i])

        for i in range(len(attn_1)):

            feature_1_map[i] = Image.fromarray(feature_1_map[i])
            align_feature_2_map[i] = Image.fromarray(align_feature_2_map[i])
            feature_2_map[i] = Image.fromarray(feature_2_map[i])
            align_feature_1_map[i] = Image.fromarray(align_feature_1_map[i])
        for i in range(len(den_scales_1_map)):
            den_scales_1_map[i]= Image.fromarray(den_scales_1_map[i])
            gt_den_scales_1_map[i]= Image.fromarray(gt_den_scales_1_map[i])
            den_scales_2_map[i]= Image.fromarray(den_scales_2_map[i])
            gt_den_scales_2_map[i]= Image.fromarray(gt_den_scales_2_map[i])
            
            mask_out_scales_1_map[i]= Image.fromarray(mask_out_scales_1_map[i])
            mask_in_scales_1_map[i]= Image.fromarray(mask_in_scales_1_map[i])
            gt_mask_out_scales_1_map[i]= Image.fromarray(gt_mask_out_scales_1_map[i])
            gt_mask_in_scales_1_map[i]= Image.fromarray(gt_mask_in_scales_1_map[i])

            den_prob_map_1[i] = Image.fromarray(den_prob_map_1[i])
            den_prob_map_2[i] = Image.fromarray(den_prob_map_2[i])
            io_prob_map_1[i] = Image.fromarray(io_prob_map_1[i])
            io_prob_map_2[i] = Image.fromarray(io_prob_map_2[i])




        # for i, box in enumerate(boxes, 0):
        #     wh_LeftTop = (box[0], box[1])
        #     wh_RightBottom = (box[2], box[3])
        #     cv2.rectangle(binar_color_map, wh_LeftTop, wh_RightBottom, point_color, thickness, lineType)
        #     cv2.rectangle(pil_input, wh_LeftTop, wh_RightBottom, point_color, thickness, lineType)

        pil_input0 = Image.fromarray(pil_input0)
        pil_input1 = Image.fromarray(pil_input1)
        den0_map = Image.fromarray(cv2.cvtColor(den0_map, cv2.COLOR_BGR2RGB))
        den1_map = Image.fromarray(cv2.cvtColor(den1_map, cv2.COLOR_BGR2RGB))
        out_map = Image.fromarray(cv2.cvtColor(out_map, cv2.COLOR_BGR2RGB))
        in_map = Image.fromarray(cv2.cvtColor(in_map, cv2.COLOR_BGR2RGB))
        conf_map0 = Image.fromarray(cv2.cvtColor(conf_map0, cv2.COLOR_BGR2RGB))
        # gt_conf_map0 = Image.fromarray(cv2.cvtColor(gt_conf_map0, cv2.COLOR_BGR2RGB))
        conf_map1 = Image.fromarray(cv2.cvtColor(conf_map1, cv2.COLOR_BGR2RGB))
        # gt_conf_map1 = Image.fromarray(cv2.cvtColor(gt_conf_map1, cv2.COLOR_BGR2RGB))



        # pil_output0 = Image.fromarray(cv2.cvtColor(pred_color_map0, cv2.COLOR_BGR2RGB))
        # pil_gt0 = Image.fromarray(cv2.cvtColor(gt_color_map0, cv2.COLOR_BGR2RGB))
        # pil_output1 = Image.fromarray(cv2.cvtColor(pred_color_map1, cv2.COLOR_BGR2RGB))
        # pil_gt1 = Image.fromarray(cv2.cvtColor(gt_color_map1, cv2.COLOR_BGR2RGB))
        # pil_maskout = Image.fromarray(cv2.cvtColor(pred_out_color, cv2.COLOR_BGR2RGB))
        # pil_gtmaskout = Image.fromarray(cv2.cvtColor(gt_out_color, cv2.COLOR_BGR2RGB))
        # pil_maskin = Image.fromarray(cv2.cvtColor(pred_in_color, cv2.COLOR_BGR2RGB))
        # pil_gtmaskin = Image.fromarray(cv2.cvtColor(gt_in_color, cv2.COLOR_BGR2RGB))

        # pil_attn_1 = Image.fromarray(cv2.cvtColor(attn_map_1, cv2.COLOR_BGR2RGB))
        # pil_attn_2 = Image.fromarray(cv2.cvtColor(attn_map_2, cv2.COLOR_BGR2RGB))

        black_map = np.zeros_like(pil_input0)
        black_map = Image.fromarray(black_map)

        

        # imgs = [pil_input0,  f_flow_map[0], f_flow_map[1], f_flow_map[2], \
        #         feature_1_map[0], feature_1_map[1], feature_1_map[2], feature_1_map[3],\
        #         align_feature_2_map[0], align_feature_2_map[1], align_feature_2_map[2], align_feature_2_map[3],\
        #         den_scales_1_map[2],den_scales_1_map[1],den_scales_1_map[0],black_map,\
        #         gt_den_scales_1_map[2],gt_den_scales_1_map[1],gt_den_scales_1_map[0],black_map,\
        #         pil_gt0, pil_output0, pil_gtmaskout, pil_maskout, \
                
                
        #         pil_input1, b_flow_map[0], b_flow_map[1], b_flow_map[2], \
        #         feature_2_map[0], feature_2_map[1], feature_2_map[2], feature_2_map[3], \
        #         align_feature_1_map[0], align_feature_1_map[1], align_feature_1_map[2],align_feature_1_map[3], \
        #         den_scales_2_map[2],den_scales_2_map[1],den_scales_2_map[0],black_map,\
        #         gt_den_scales_2_map[2],gt_den_scales_2_map[1],gt_den_scales_2_map[0],black_map,\
        #         pil_gt1, pil_output1, pil_gtmaskin, pil_maskin]
        imgs = [pil_input0,  den0_map, out_map,\
                black_map, conf_map0, black_map,\
                den_prob_map_1[2],den_prob_map_1[1],den_prob_map_1[0],\
                io_prob_map_1[2],io_prob_map_1[1],io_prob_map_1[0],\
                
                f_flow_map[2], f_flow_map[1], f_flow_map[0], \
                feature_1_map[2], feature_1_map[1], feature_1_map[0],\
                align_feature_2_map[2], align_feature_2_map[1], align_feature_2_map[0],\
                den_scales_1_map[2],den_scales_1_map[1],den_scales_1_map[0],\
                gt_den_scales_1_map[2],gt_den_scales_1_map[1],gt_den_scales_1_map[0],\
                mask_out_scales_1_map[2],mask_out_scales_1_map[1],mask_out_scales_1_map[0],\
                gt_mask_out_scales_1_map[2],gt_mask_out_scales_1_map[1],gt_mask_out_scales_1_map[0],\
                
                
                
                pil_input1,  den1_map, in_map,\
                black_map, conf_map1, black_map,\
                den_prob_map_2[2],den_prob_map_2[1],den_prob_map_2[0],\
                io_prob_map_2[2],io_prob_map_2[1],io_prob_map_2[0],\
                b_flow_map[2], b_flow_map[1], b_flow_map[0], \
                feature_2_map[2], feature_2_map[1], feature_2_map[0], \
                align_feature_1_map[2], align_feature_1_map[1], align_feature_1_map[0], \
                den_scales_2_map[2],den_scales_2_map[1],den_scales_2_map[0],\
                gt_den_scales_2_map[2],gt_den_scales_2_map[1],gt_den_scales_2_map[0],\
                mask_in_scales_1_map[2],mask_in_scales_1_map[1],mask_in_scales_1_map[0],\
                gt_mask_in_scales_1_map[2],gt_mask_in_scales_1_map[1],gt_mask_in_scales_1_map[0]]


        w_num , h_num = 3, 22

        

        target_shape = (w_num * (UNIT_W + 10), h_num * (UNIT_H + 10))
        target = Image.new('RGB', target_shape)
        count = 0
        for img in imgs:
            x, y = int(count%w_num) * (UNIT_W + 10), int(count // w_num) * (UNIT_H + 10)  # 左上角坐标，从左到右递增
            target.paste(img, (x, y, x + UNIT_W, y + UNIT_H))
            count+=1


        if cfg.mode == 'test':
           
            try:    
                dir = os.path.join(cfg.model_path.replace('exp', cfg.output_dir).replace(cfg.model_path.split('/')[-1],os.path.basename(cfg.model_path).split('_')[3]),scene_name.split('/')[-1])
            except:
                dir = './'
        else:
            dir = os.path.join(exp_path, exp_name, 'vis')
        if not os.path.isdir(dir):
            os.makedirs(dir)

        target.resize((w_num*50, h_num*50))
        target.save(os.path.join(dir,f'{iter}_{batch}_den.jpg'.format()))

        # target.save(os.path.join(exp_path,'onlymean_offset','{}_den.jpg'.format(iter)))



        

    


# -

# def vis_results_more(exp_name, epoch, writer, restore, img, pred_map, gt_map, binar_map,threshold_matrix, pred_boxes, gt_boxes):

#     pil_to_tensor = standard_transforms.ToTensor()

#     x = []
#     y = []

#     for idx, tensor in enumerate(zip(img.cpu().data, pred_map, gt_map, binar_map, threshold_matrix)):
#         if idx > 1:  # show only one group
#             break

#         pil_input = restore(tensor[0])
#         pred_color_map = cv2.applyColorMap((255 * tensor[1] / (tensor[2].max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
#         gt_color_map = cv2.applyColorMap((255 * tensor[2] / (tensor[2].max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
#         binar_color_map = cv2.applyColorMap((255 * tensor[3] ).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
#         threshold_color_map = cv2.applyColorMap((255 * tensor[4] / (tensor[2].max()  + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)

#         point_color = (0, 255, 0)  # BGR
#         thickness = 1
#         lineType = 4
#         pil_input = np.array(pil_input)

#         for i, box in enumerate(pred_boxes, 0):
#             wh_LeftTop = (box[0], box[1])
#             wh_RightBottom = (box[2], box[3])
#             # print(wh_LeftTop, wh_RightBottom)
#             cv2.rectangle(binar_color_map, wh_LeftTop, wh_RightBottom, point_color, thickness, lineType)
#             cv2.rectangle(pil_input, wh_LeftTop, wh_RightBottom, point_color, thickness, lineType)
#         point_color = (255, 0, 0)  # BGR

#         for i, box in enumerate(gt_boxes, 0):
#             wh_LeftTop = (box[0], box[1])
#             wh_RightBottom = (box[2], box[3])
#             cv2.rectangle(pil_input, wh_LeftTop, wh_RightBottom, point_color, thickness, lineType)

#         pil_input = Image.fromarray(pil_input)
#         pil_label = Image.fromarray(cv2.cvtColor(gt_color_map, cv2.COLOR_BGR2RGB))
#         pil_output = Image.fromarray(cv2.cvtColor(pred_color_map, cv2.COLOR_BGR2RGB))
#         pil_binar = Image.fromarray(cv2.cvtColor(binar_color_map, cv2.COLOR_BGR2RGB))

#         pil_threshold = Image.fromarray(cv2.cvtColor(threshold_color_map, cv2.COLOR_BGR2RGB))


#         x.extend([pil_to_tensor(pil_input.convert('RGB')), pil_to_tensor(pil_label.convert('RGB')),
#                   pil_to_tensor(pil_output.convert('RGB')), pil_to_tensor(pil_binar.convert('RGB')),
#                   pil_to_tensor(pil_threshold.convert('RGB'))])

#     x = torch.stack(x, 0)
#     x = vutils.make_grid(x, nrow=3, padding=5)
#     x = (x.numpy() * 255).astype(np.uint8)

#     writer.add_image(exp_name + '_epoch_' + str(epoch + 1), x)

# def vis_results(exp_name, epoch, writer, restore, img, pred_map, gt_map,binar_map,boxes):#, flow):

#     pil_to_tensor = standard_transforms.ToTensor()

#     x = []
#     y = []

#     for idx, tensor in enumerate(zip(img.cpu().data, pred_map, gt_map,binar_map)):
#         if idx>1:# show only one group
#             break

#         pil_input = restore(tensor[0])
#         pred_color_map = cv2.applyColorMap((255*tensor[1]/(tensor[2].max()+1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
#         gt_color_map = cv2.applyColorMap((255*tensor[2]/(tensor[2].max()+1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
#         binar_color_map = cv2.applyColorMap((255*tensor[3]/(tensor[2].max()+1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)

#         point_color = (0, 255, 0)  # BGR
#         thickness = 1
#         lineType = 4
#         pil_input = np.array(pil_input)
#         # print(pil_input, binar_color_map)
#         for i, box in enumerate(boxes, 0):
#             wh_LeftTop = (box[0], box[1])
#             wh_RightBottom = (box[0] + box[2], box[1] + box[3])
#             # print(wh_LeftTop, wh_RightBottom)
#             cv2.rectangle(binar_color_map, wh_LeftTop, wh_RightBottom, point_color, thickness, lineType)
#             cv2.rectangle(pil_input,       wh_LeftTop, wh_RightBottom, point_color, thickness, lineType)

#         pil_input =Image.fromarray(pil_input)
#         pil_label = Image.fromarray(cv2.cvtColor(gt_color_map,cv2.COLOR_BGR2RGB))
#         pil_output = Image.fromarray(cv2.cvtColor(pred_color_map,cv2.COLOR_BGR2RGB))
#         pil_binar = Image.fromarray(cv2.cvtColor(binar_color_map, cv2.COLOR_BGR2RGB))
#         x.extend([pil_to_tensor(pil_input.convert('RGB')), pil_to_tensor(pil_label.convert('RGB')),
#                   pil_to_tensor(pil_output.convert('RGB')),pil_to_tensor(pil_binar.convert('RGB'))])
#         # pdb.set_trace()  sum(sum(flow[0].cpu().data.numpy().transpose((1,2,0))[:,:,0]))
#         # flow = flow[0].cpu().data.numpy().transpose((1,2,0))
#         # flow0 = cv2.applyColorMap((255*flow[:,:,0]/(flow[:,:,0].max()+1e-10)).astype(np.uint8).squeeze(),cv2.COLORMAP_JET)
#         # flow1 = cv2.applyColorMap((255*flow[:,:,1]/(flow[:,:,1].max()+1e-10)).astype(np.uint8).squeeze(),cv2.COLORMAP_JET)
#         # flow2 = cv2.applyColorMap((255*flow[:,:,2]/(flow[:,:,2].max()+1e-10)).astype(np.uint8).squeeze(),cv2.COLORMAP_JET)
#         # flow0 = Image.fromarray(cv2.cvtColor(flow0,cv2.COLOR_BGR2RGB))
#         # flow1 = Image.fromarray(cv2.cvtColor(flow1,cv2.COLOR_BGR2RGB))
#         # flow2 = Image.fromarray(cv2.cvtColor(flow2,cv2.COLOR_BGR2RGB))
#         # y.extend([pil_to_tensor(flow0.convert('RGB')), pil_to_tensor(flow1.convert('RGB')), pil_to_tensor(flow2.convert('RGB'))])


#     x = torch.stack(x, 0)
#     x = vutils.make_grid(x, nrow=4, padding=5)
#     x = (x.numpy()*255).astype(np.uint8)

#     # y = torch.stack(y,0)
#     # y = vutils.make_grid(y,nrow=3,padding=5)
#     # y = (y.numpy()*255).astype(np.uint8)

#     # x = np.concatenate((x,y),axis=1)
#     writer.add_image(exp_name + '_epoch_' + str(epoch+1), x)


# def print_NWPU_summary(trainer, scores):
#     f1m_l, ap_l, ar_l, mae, mse, nae, loss = scores
#     train_record = trainer.train_record
#     with open(trainer.log_txt, 'a') as f:
#         f.write('='*15 + '+'*15 + '='*15 + '\n')
#         f.write(str(trainer.epoch) + '\n\n')

#         f.write('  [F1 %.4f Pre %.4f Rec %.4f ] [mae %.4f mse %.4f nae %.4f] [val loss %.4f]\n\n' % (f1m_l, ap_l, ar_l,mae, mse, nae,loss))

#         f.write('='*15 + '+'*15 + '='*15 + '\n\n')

#     print( '='*50 )
#     print( trainer.exp_name )
#     print( '    '+ '-'*20 )
#     print( '  [F1 %.4f Pre %.4f Rec %.4f] [mae %.2f mse %.2f], [val loss %.4f]'\
#             % (f1m_l, ap_l, ar_l, mae, mse, loss) )
#     print( '    '+ '-'*20 )
#     print( '[best] [model: %s] , [F1 %.4f Pre %.4f Rec %.4f] [mae %.2f], [mse %.2f], [nae %.4f]' % (train_record['best_model_name'], \
#                                                         train_record['best_F1'], \
#                                                         train_record['best_Pre'], \
#                                                         train_record['best_Rec'],\
#                                                         train_record['best_mae'],\
#                                                         train_record['best_mse'],\
#                                                         train_record['best_nae']) )
#     print( '='*50 )  

def print_NWPU_summary_det(trainer, scores):
    train_record = trainer.train_record
    with open(trainer.log_txt, 'a') as f:
        f.write('='*15 + '+'*15 + '='*15 + '\n')
        f.write(str(trainer.epoch) + '\n\n')
        f.write('  [')
        for key, data in scores.items():
            f.write(' ' +key+  ' %.3f'% data)
        f.write('\n\n')
        f.write('='*15 + '+'*15 + '='*15 + '\n\n')

    print( '='*50 )
    print( trainer.exp_name )
    print( '    '+ '-'*20 )
    content = '  ['
    for key, data in scores.items():
        if isinstance(data,str):
            content +=(' ' + key + ' %s' % data)
        else:
            content += (' ' + key + ' %.3f' % data)
    content += ']'
    print( content)
    print( '    '+ '-'*20 )
    best_str = '[best]'
    for key, data in train_record.items():
        best_str += (' [' + key +' %s'% data + ']')
    print( best_str)
    print( '='*50 )

def update_model(trainer, scores, val=False):
    train_record = trainer.train_record
    if val:
        log_file = trainer.log_txt
        epoch = trainer.epoch
        snapshot_name = 'ep_%d_iter_%d'% (epoch,trainer.i_tb)
        for key, data in scores.items():
            snapshot_name+= ('_'+ key+'_%.3f'%data)
        # snapshot_name = 'ep_%d_F1_%.3f_Pre_%.3f_Rec_%.3f_mae_%.1f_mse_%.1f' % (epoch + 1, F1, Pre, Rec, mae, mse)

        for key, data in  scores.items():
            print(key,data)
            best = False
            if data<train_record[key] :
                best = True
                train_record['best_model_name'] = snapshot_name
                to_saved_weight = trainer.net.state_dict()

                torch.save(to_saved_weight, os.path.join(trainer.exp_path, trainer.exp_name, snapshot_name + '.pth'))
            if (best == True) and(log_file is not None):
                logger_txt(log_file,epoch,trainer.i_tb,scores)

            if data < train_record[key]:
                train_record[key] = data
    latest_state = {'train_record':train_record, 'net':trainer.net.state_dict(), 'optimizer':trainer.optimizer.state_dict(),'epoch': trainer.epoch, 'i_tb':trainer.i_tb,\
                    'num_iters':trainer.num_iters, 'exp_path':trainer.exp_path, 'exp_name':trainer.exp_name, 'cfg':trainer.cfg}
    torch.save(latest_state,os.path.join(trainer.exp_path, trainer.exp_name, 'latest_state.pth'))

    return train_record


def copy_cur_env(work_dir, dst_dir, exception):

    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    for filename in os.listdir(work_dir):

        file = os.path.join(work_dir,filename)
        dst_file = os.path.join(dst_dir,filename)

        if os.path.isdir(file) and filename not in exception:
            shutil.copytree(file, dst_file)
        elif os.path.isfile(file):
            shutil.copyfile(file,dst_file)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.cur_val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, cur_val):
        self.cur_val = cur_val
        self.sum += cur_val
        self.count += 1
        self.avg = self.sum / self.count


class AverageCategoryMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self,num_class):
        self.num_class = num_class
        self.reset()

    def reset(self):
        self.cur_val = np.zeros(self.num_class)
        self.sum = np.zeros(self.num_class)


    def update(self, cur_val):
        self.cur_val = cur_val
        self.sum += cur_val


# class AverageCategoryMeter(object):
#     """Computes and stores the average and current value"""
#
#     def __init__(self,num_class):
#         self.num_class = num_class
#         self.reset()
#
#     def reset(self):
#         self.cur_val = np.zeros(self.num_class)
#         self.avg = np.zeros(self.num_class)
#         self.sum = np.zeros(self.num_class)
#         self.count = np.zeros(self.num_class)
#
#     def update(self, cur_val, class_id):
#         self.cur_val[class_id] = cur_val
#         self.sum[class_id] += cur_val
#         self.count[class_id] += 1
#         self.avg[class_id] = self.sum[class_id] / self.count[class_id]


class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def vis_results_img(img, pred, restore):
    # pdb.set_trace()
    img = img.cpu()
    pred = pred.cpu().numpy()
    pil_input = restore(img)
    pred_color_map = cv2.applyColorMap(
        (255*pred / (pred.max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
    pil_output = Image.fromarray(
        cv2.cvtColor(pred_color_map, cv2.COLOR_BGR2RGB))
    x = []
    pil_to_tensor = standard_transforms.ToTensor()
    x.extend([pil_to_tensor(pil_input.convert('RGB')),
              pil_to_tensor(pil_output.convert('RGB'))])
    x = torch.stack(x, 0)
    x = vutils.make_grid(x, nrow=3, padding=5)
    x = (x.numpy() * 255).astype(np.uint8)

    # pdb.set_trace()
    return x

def make_matching_plot_fast(image0, image1, kpts0, kpts1, mkpts0,
                            mkpts1, color, text, path=None,
                            show_keypoints=False, margin=10,
                            opencv_display=False, opencv_title='',
                            small_text = [], restore_transform=None,
                            id0=None,id1=None
                            ):

    image0 = np.array(restore_transform(image0))
    image1 = np.array(restore_transform(image1))
    image0 = cv2.cvtColor(image0, cv2.COLOR_RGB2BGR)
    image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)
    H0, W0, C = image0.shape
    H1, W1, C = image1.shape
    H, W = max(H0, H1), W0 + W1 + margin

    out = 255*np.ones((H, W, C), np.uint8)
    out[:H0, :W0,:] = image0
    out[:H1, W0+margin:,:] = image1
    # out = np.stack([out]*3, -1)
    # import pdb
    # pdb.set_trace()
    out_by_point = out.copy()
    point_r_value = 15
    thickness = 3
    white = (255, 255, 255)
    green = (0, 255, 0)
    red = (0, 0, 255)
    blue = (255, 0, 0)
    if show_keypoints:
        kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
        for x, y in kpts0:
            cv2.circle(out, (x, y), point_r_value, red, thickness, lineType=cv2.LINE_AA)
            cv2.circle(out, (x, y), 3, white, -1, lineType=cv2.LINE_AA)

            cv2.circle(out_by_point, (x, y), point_r_value, red, thickness, lineType=cv2.LINE_AA)

        for x, y in kpts1:
            cv2.circle(out, (x + margin + W0, y), point_r_value, red, thickness,
                       lineType=cv2.LINE_AA)
            cv2.circle(out, (x + margin + W0, y), 3, white, -1, lineType=cv2.LINE_AA)

            cv2.circle(out_by_point, (x + margin + W0, y), point_r_value, blue, thickness,
                       lineType=cv2.LINE_AA)

        if id0 is not  None:
            for i, (id, centroid) in enumerate(zip(id0, kpts0)):
                cv2.putText(out, str(id), (centroid[0],centroid[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        if id1 is not None:
            for i, (id, centroid) in enumerate(zip(id1, kpts1)):
                cv2.putText(out, str(id), (centroid[0]+margin+W0, centroid[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
    color = (np.array(color[:, :3])*255).astype(int)[:, ::-1]

    for (x0, y0), (x1, y1), c in zip(mkpts0, mkpts1, color):
        c = c.tolist()
        cv2.line(out, (x0, y0), (x1 + margin + W0, y1),
                 color=c, thickness=1, lineType=cv2.LINE_AA)
        # display line end-points as circles
        cv2.circle(out, (x0, y0), point_r_value, green, thickness, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + W0, y1), point_r_value, green, thickness,
                   lineType=cv2.LINE_AA)

        cv2.circle(out_by_point, (x0, y0), point_r_value, green, thickness, lineType=cv2.LINE_AA)
        cv2.circle(out_by_point, (x1 + margin + W0, y1), point_r_value, green, thickness,
                   lineType=cv2.LINE_AA)

    # Ht = int(H*30 / 480)  # text height
    # txt_color_fg = (255, 255, 255)
    # txt_color_bg = (0, 0, 0)
    # for i, t in enumerate(text):
    #     cv2.putText(out, t, (10, Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
    #                 H*1.0/480, txt_color_bg, 2, cv2.LINE_AA)
    #     cv2.putText(out, t, (10, Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
    #                 H*1.0/480, txt_color_fg, 1, cv2.LINE_AA)
    #     cv2.putText(out_by_point, t, (10, Ht * (i + 1)), cv2.FONT_HERSHEY_DUPLEX,
    #             H * 1.0 / 480, txt_color_fg, 1, cv2.LINE_AA)
    if path is not None:
        cv2.imwrite(str(path), out)
        cv2.imwrite(str('point_'+path), out_by_point)
    if opencv_display:
        cv2.imshow(opencv_title, out)
        cv2.waitKey(1)

    return out,out_by_point
