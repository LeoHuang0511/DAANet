#!/usr/bin/env python
# coding: utf-8

import os.path as osp
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data

from torchvision.ops.boxes import clip_boxes_to_image
from PIL import Image
import re
import cv2
from .shift_transform import CutMixShift




# +
class Dataset(data.Dataset):
    """
    Dataset class.
    """
    def __init__(self, txt_path, base_path, cfg, main_transform=None,img_transform=None,train=True, datasetname='Empty'):
        self.base_path = base_path
        self.bboxes = defaultdict(list)
        self.imgs_path = []
        self.labels = []
        self.datasetname = datasetname
        if train:
            with open(osp.join(base_path, txt_path), 'r') as txt:
                scene_names = txt.readlines()
        else:
            scene_names = txt_path             # for val and test

        for i in scene_names:
            
            if datasetname == "CARLA":
                img_path, label= CARLA_ImgPath_and_Target(base_path,i.strip())
            elif datasetname == "HT21":
                img_path, label= HT21_ImgPath_and_Target(base_path,i.strip())
            elif datasetname == "SENSE":
                img_path, label= SENSE_ImgPath_and_Target(base_path,i.strip())
            else:
                raise NotImplementedError
            self.imgs_path+=img_path
            self.labels +=label

        self.is_train = train
        self.main_transforms = main_transform
        self.img_transforms = img_transform

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):

#         img_rgb = Image.open(self.imgs_path[index].replace('_resize.h5','.jpg'))
        # img = cv2.imread(self.imgs_path[index].replace('_resize.h5','.jpg'))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        # img = read_LAB_image(self.imgs_path[index])
        img = Image.open(self.imgs_path[index])
        if img.mode != 'RGB':
            img=img.convert('RGB')
#         if img.mode != 'RGB':
#             img=img.convert('RGB')

        target = self.labels[index].copy()

        if self.main_transforms != None:
            img, target = self.main_transforms(img, target)
        # img_rgb = cv2.cvtColor(np.array(img.copy()), cv2.COLOR_LAB2RGB)
        # img_rgb = torch.from_numpy(img_rgb).float().permute(2,0,1)
        if self.img_transforms != None:
            img = self.img_transforms(img)

        return  img, target


# -

class TestDataset(data.Dataset):
    """
    Dataset class.
    """
    def __init__(self,scene_name, base_path, cfg, main_transform=None, img_transform=None, interval=1, target=True, datasetname='Empty'):
        self.base_path = base_path
        self.target = target
        self.cfg = cfg

        if self.target:
            
            if datasetname == 'CARLA':
                self.imgs_path, self.label = CARLA_ImgPath_and_Target(self.base_path, scene_name)
            elif datasetname == 'HT21':
                self.imgs_path, self.label = HT21_ImgPath_and_Target(self.base_path, scene_name)
            elif datasetname == "SENSE":
                self.imgs_path, self.label= SENSE_ImgPath_and_Target(self.base_path, scene_name)
            else:
                raise NotImplementedError
        else:
            if datasetname == 'HT21':
                self.imgs_path  = self.generate_imgPath_label(scene_name)
            else:
                raise NotImplementedError

        
        self.interval =interval

        self.main_transforms = main_transform
        self.img_transforms = img_transform
        self.length =  len(self.imgs_path)
    def __len__(self):
        return len(self.imgs_path) - self.interval


    def __getitem__(self, index):

        index1 = index
        index2 = index + self.interval//2

        


        # img1 = read_LAB_image(self.imgs_path[index1])
        # img2 = read_LAB_image(self.imgs_path[index2])
        # img3 = read_LAB_image(self.imgs_path[index3])
        img1 = Image.open(self.imgs_path[index1])
        img2 = Image.open(self.imgs_path[index2])

        if img1.mode != 'RGB':
            img1=img1.convert('RGB')
        if img2.mode != 'RGB':
            img2 = img2.convert('RGB')

        




        if self.img_transforms != None:
            img1 = self.img_transforms(img1)
            img2 = self.img_transforms(img2)
 
        

        if self.target:
            target1 = self.label[index1]
            target2 = self.label[index2]
            return  [img1,img2], [target1,target2]
            

        return [img1,img2], None

  
    def generate_imgPath_label(self, i):

        img_path = []
        root = osp.join(self.base_path, i +'/img1')
        img_ids = os.listdir(root)
        img_ids.sort(key=self.myc)


        for img_id in img_ids:
            img_id = img_id.strip()
            single_path = osp.join(root, img_id)
            img_path.append(single_path)

        return img_path
    def myc(self, string):
        p = re.compile("\d+")
        return int(p.findall(string)[0])

class ShiftPretrainDataset(data.Dataset):
    """
    Dataset class.
    """
    def __init__(self, txt_path, base_path, cfg, main_transform=None,img_transform=None,train=True, datasetname='Empty'):
        self.base_path = base_path
        self.bboxes = defaultdict(list)
        self.imgs_path = []
        self.labels = []
        self.datasetname = datasetname
        self.cfg = cfg

        with open(osp.join(base_path, txt_path), 'r') as txt:
            scene_names = txt.readlines()

        for i in scene_names:
            
            if datasetname == "CARLA":
                img_path, label= CARLA_ImgPath_and_Target(base_path,i.strip())
            elif datasetname == "HT21":
                img_path, label= HT21_ImgPath_and_Target(base_path,i.strip())
            elif datasetname == "SENSE":
                img_path, label= SENSE_ImgPath_and_Target(base_path,i.strip())


            else:
                raise NotImplementedError
            self.imgs_path+=img_path
            self.labels +=label

        self.is_train = train
        self.main_transforms = main_transform
        self.img_transforms = img_transform
        self.cutmix_shift = CutMixShift(self.cfg)

        self.index_sequence = np.arange(len(self.imgs_path))

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):

#         img_rgb = Image.open(self.imgs_path[index].replace('_resize.h5','.jpg'))
        # img = cv2.imread(self.imgs_path[index].replace('_resize.h5','.jpg'))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        result = np.random.choice(self.index_sequence, size=3, replace=False)

        

        img1 = read_LAB_image(self.imgs_path[index])
        img2 = read_LAB_image(self.imgs_path[result[0]])
        img3 = read_LAB_image(self.imgs_path[result[1]])
        img4 = read_LAB_image(self.imgs_path[result[2]])

        target1 = self.labels[index].copy()
        target2 = self.labels[result[0]].copy()
        target3 = self.labels[result[1]].copy()
        target4 = self.labels[result[2]].copy()

        shift_img1, shift_img2, shift_target1, shift_target2 = self.cutmix_shift(img1, img2, img3, img4, target1, target2, target3, target4)



        if self.main_transforms != None:
            shift_img1, shift_target1 = self.main_transforms(shift_img1, shift_target1)
            shift_img2, shift_target2 = self.main_transforms(shift_img2, shift_target2)
                                                             

        if self.img_transforms != None:
            shift_img1 = self.img_transforms(shift_img1)
            shift_img2 = self.img_transforms(shift_img2)

        return  torch.cat([shift_img1[None,:], shift_img2[None,:]],dim=0), [shift_target1, shift_target2]



def CARLA_ImgPath_and_Target(base_path,i):
    img_path = []
    labels=[]
    root  = osp.join(base_path, i + '/img1')
    img_ids = os.listdir(root)
    img_ids.sort()
    gts = defaultdict(list)
    with open(osp.join(root.replace('img1', 'gt'), 'gt.txt'), 'r') as f:
        lines = f.readlines()
        for lin in lines:
            lin_list = [float(i) for i in lin.rstrip().split(',')]
            ind = int(lin_list[0])
            gts[ind].append(lin_list)

    for img_id in img_ids:
        img_id = img_id.strip()
        single_path = osp.join(root, img_id)
        annotation  = gts[int(img_id.split('.')[0].replace('_resize',''))]
        annotation = torch.tensor(annotation,dtype=torch.float32)
        box = annotation[:,2:6]
        points =   box[:,0:2] + box[:,2:4]/2

        sigma = torch.min(box[:,2:4], 1)[0] / 2.
        ids = annotation[:,1].long()
        img_path.append(single_path)

        labels.append({'scene_name':i,'frame':int(img_id.split('.')[0].replace('_resize','')), 'person_id':ids, 'points':points,'sigma':sigma})
    return img_path, labels

def HT21_ImgPath_and_Target(base_path,i):
    img_path = []
    labels=[]
    root  = osp.join(base_path, i + '/img1')
    img_ids = os.listdir(root)
    img_ids.sort()
    gts = defaultdict(list)
    with open(osp.join(root.replace('img1', 'gt'), 'gt.txt'), 'r') as f:
        lines = f.readlines()
        for lin in lines:
            lin_list = [float(i) for i in lin.rstrip().split(',')]
            ind = int(lin_list[0])
            gts[ind].append(lin_list)

    for img_id in img_ids:
        img_id = img_id.strip()
        single_path = osp.join(root, img_id)
        annotation  = gts[int(img_id.split('.')[0].replace('_resize',''))]
        annotation = torch.tensor(annotation,dtype=torch.float32)
        box = annotation[:,2:6]
        points =   box[:,0:2] + box[:,2:4]/2

        sigma = torch.min(box[:,2:4], 1)[0] / 2.
        ids = annotation[:,1].long()
        img_path.append(single_path)

        labels.append({'scene_name':i,'frame':int(img_id.split('.')[0].replace('_resize','')), 'person_id':ids, 'points':points,'sigma':sigma})
    return img_path, labels


def SENSE_ImgPath_and_Target(base_path,i):
    img_path = []
    labels=[]
    root  = osp.join(base_path, 'video_ori', i )
    img_ids = os.listdir(root)
    img_ids.sort()
    gts = defaultdict(list)
    with open(root.replace('video_ori', 'label_list_all')+'.txt', 'r') as f: #label_list_all_rmInvalid
        lines = f.readlines()
        for lin in lines:
            lin_list = [i for i in lin.rstrip().split(' ')]
            ind = lin_list[0]
            lin_list = [float(i) for i in lin_list[3:] if i != '']
            assert len(lin_list) % 7 == 0
            gts[ind] = lin_list

    for img_id in img_ids:
        img_id = img_id.strip()
        single_path = osp.join(root, img_id)
        label = gts[img_id]
        box_and_point = torch.tensor(label).view(-1, 7).contiguous()

        points = box_and_point[:, 4:6].float()
        ids = (box_and_point[:, 6]).long()

        if ids.size(0)>0:
            sigma = 0.6*torch.stack([(box_and_point[:,2]-box_and_point[:,0])/2,(box_and_point[:,3]-box_and_point[:,1])/2],1).min(1)[0]  #torch.sqrt(((box_and_point[:,2]-box_and_point[:,0])/2)**2 + ((box_and_point[:,3]-box_and_point[:,1])/2)**2)
        else:
            sigma = torch.tensor([])
        img_path.append(single_path)

        labels.append({'scene_name':i,'frame':int(img_id.split('.')[0].replace('_resize','')), 'person_id':ids, 'points':points, 'sigma':sigma})
    return img_path, labels

