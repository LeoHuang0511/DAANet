# -*- coding: utf-8 -*-

import os
import torch
import torch.nn.functional as F
from importlib import import_module
import misc.transforms as own_transforms
from  misc.transforms import  check_image
import torchvision.transforms as standard_transforms
from . import dataset
from . import setting
from . import samplers
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from config import  cfg

# +
import random
class train_pair_transform(object):
    def __init__(self,cfg_data, check_dim = True):
        self.cfg_data = cfg_data
        self.pair_flag = 0
        self.scale_factor = 1
        self.last_cw_ch =(0,0)
        self.crop_left = (0,0)
        self.last_crop_left = (0, 0)
        self.rate_range = (0.8,1.2)
        self.resize_and_crop= own_transforms.RandomCrop( cfg_data.TRAIN_SIZE)
        self.scale_to_setting = own_transforms.ScaleByRateWithMin(cfg_data.TRAIN_SIZE[1], cfg_data.TRAIN_SIZE[0])

        self.flip_flag = 0
        self.horizontal_flip = own_transforms.RandomHorizontallyFlip()

        self.last_frame_size = (0,0)

        self.check_dim = check_dim
    def __call__(self,img,target):
        import numpy as np
        w_ori, h_ori = img.size
        if self.pair_flag == 1 and self.check_dim:  # make sure two frames are with the same shape
            assert self.last_frame_size == (w_ori,w_ori)
            # self.last_frame_size = (w_ori, w_ori)
          # make sure the img size is large than we needed
        
        if self.pair_flag % 2 == 0:
            self.scale_factor = random.uniform(self.rate_range[0], self.rate_range[1])

            self.c_h,self.c_w = int(self.cfg_data.TRAIN_SIZE[0]/self.scale_factor), int(self.cfg_data.TRAIN_SIZE[1]/self.scale_factor)
            img, target = check_image(img, target, (self.c_h,self.c_w))
            w, h = img.size
            self.last_cw_ch = (self.c_w,self.c_h)
            self.pair_flag = 0
            self.last_frame_size = (w_ori, w_ori)

            x1 = random.randint(0, w - self.c_w)

            y1 = random.randint(0, h - self.c_h)
            self.last_crop_left = (x1,y1)
            self.flip_flag = round(random.random())
        else:
            img, target = check_image(img, target, (self.c_h,self.c_w))

#         if self.pair_flag % 2 == 1:
#         img, target = check_image(img, target, (self.c_h,self.c_w))
            
#             if self.check_dim:
#                 x1 = max(0, int(self.last_crop_left[0] + (self.last_cw_ch[0]-self.c_w)))
#                 y1 = max(0, int(self.last_crop_left[1] + (self.last_cw_ch[1]-self.c_h)))
#             else:   # for pre_training on other dataset
#                 x1 = random.randint(0, w - self.c_w)
#                 y1 = random.randint(0, h - self.c_h)
            
#         self.crop_left = (x1, y1)

        img, target = self.resize_and_crop(img, target, self.crop_left,crop_size=(self.c_h,self.c_w))
        img, target = self.scale_to_setting(img,target)

        img, target = self.horizontal_flip(img, target, self.flip_flag)
        self.pair_flag += 1

        # assert np.array(img).sum()>0
        return img, target
    
class color_pair_transform(object):
    def __init__(self,cfg_data, check_dim = True):
        self.cfg_data = cfg_data
        self.scale_factor = 1

        self.crop_left = (0,0)

        self.rate_range = (0.8,1.2)
        self.resize_and_crop= own_transforms.RandomCrop( cfg_data.TRAIN_SIZE)
        self.scale_to_setting = own_transforms.ScaleByRateWithMin(cfg_data.TRAIN_SIZE[1], cfg_data.TRAIN_SIZE[0])

        self.flip_flag = 0
        self.horizontal_flip = own_transforms.RandomHorizontallyFlip()
        self.scene_name = ""


        self.check_dim = check_dim
    def __call__(self,img,target):
        w_ori, h_ori = img.size
#         w_ori = img.shape[1]
#         h_ori = img.shape[0]

        if self.scene_name == target["scene_name"]:
            img, target = check_image(img, target, (self.c_h,self.c_w))  # make sure the img size is large than we needed
            img, target = self.resize_and_crop(img, target, self.crop_left,crop_size=(self.c_h,self.c_w))
            img, target = self.scale_to_setting(img,target)
            img, target = self.horizontal_flip(img, target, self.flip_flag)
        else:
            self.scene_name = target["scene_name"]
            self.scale_factor = random.uniform(self.rate_range[0], self.rate_range[1])
            self.c_h,self.c_w = int(self.cfg_data.TRAIN_SIZE[0]/self.scale_factor), int(self.cfg_data.TRAIN_SIZE[1]/self.scale_factor)
            img, target = check_image(img, target, (self.c_h,self.c_w))  # make sure the img size is large than we needed
            w, h = img.size

            x1 = random.randint(0, w - self.c_w)
            y1 = random.randint(0, h - self.c_h)
            self.crop_left = (x1, y1)

            img, target = self.resize_and_crop(img, target, self.crop_left,crop_size=(self.c_h,self.c_w))
            img, target = self.scale_to_setting(img,target)

            self.flip_flag = round(random.random())
            img, target = self.horizontal_flip(img, target, self.flip_flag)

        return img, target


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    # return torch.utils.data.dataloader.default_collate(batch)
    # if len(batch) == 0:
    #     import pdb;pdb.set_trace()
    return tuple(zip(*batch))

def createTrainData(datasetname, Dataset, cfg_data):
    img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*cfg_data.MEAN_STD)
        
    ])

    main_transform = train_pair_transform(cfg_data)
    color_transform = color_pair_transform(cfg_data)
    train_set =Dataset(cfg_data.TRAIN_LST,
                                    cfg_data.DATA_PATH,
                                    main_transform=main_transform,
                                    img_transform=img_transform,
                                     train=True,
                                    datasetname=datasetname)
    color_set = Dataset(cfg_data.TRAIN_LST,
                                    cfg_data.DATA_PATH,
                                    main_transform=color_transform,
                                    img_transform=img_transform,
                                     train=True,
                                    datasetname=datasetname)

    train_sampler = samplers.CategoriesSampler(train_set.labels, frame_intervals=cfg_data.TRAIN_FRAME_INTERVALS,
                                                   n_per=cfg_data.TRAIN_BATCH_SIZE)
    color_sampler = samplers.ColorizationSampler(train_set.labels,cfg_data.TRAIN_FRAME_INTERVALS)
    
    train_loader = DataLoader(train_set, batch_sampler=train_sampler, num_workers=4, collate_fn=collate_fn, pin_memory=True)
    color_loader = DataLoader(color_set, batch_sampler=color_sampler, num_workers=1, collate_fn=collate_fn, pin_memory=True)
    print('dataset is {}, images num is {}'.format(datasetname, train_set.__len__()))

    return  train_loader, color_loader
def createValData(datasetname, Dataset, cfg_data):
    img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*cfg_data.MEAN_STD)
    ])

    val_loader = []
    with open(os.path.join( cfg_data.DATA_PATH, cfg_data.VAL_LST), 'r') as txt:
        scene_names = txt.readlines()
    for scene in scene_names:
        sub_val_dataset = Dataset([scene.strip()],
                                  cfg_data.DATA_PATH,
                                  main_transform=None,
                                  img_transform= img_transform ,
                                  train=False,
                                  datasetname=datasetname)
        sub_val_loader = DataLoader(sub_val_dataset, batch_size=cfg_data.VAL_BATCH_SIZE, num_workers=4,collate_fn=collate_fn,pin_memory=False )
        val_loader.append(sub_val_loader)

    return  val_loader
def createRestore(mean_std):
    return standard_transforms.Compose([
        own_transforms.DeNormalize(*mean_std),
        standard_transforms.ToPILImage()
    ])

def loading_data(datasetname,val_interval):
    datasetname = datasetname.upper()
    cfg_data = getattr(setting, datasetname).cfg_data

    Dataset = dataset.Dataset
    print(cfg_data)
    train_loader, color_loader = createTrainData(datasetname, Dataset, cfg_data)
#     train_loader = createTrainData(datasetname, Dataset, cfg_data)
    restore_transform = createRestore(cfg_data.MEAN_STD)

    Dataset = dataset.TestDataset
    val_loader = createValTestData(datasetname, Dataset, cfg_data,val_interval, mode ='val')


    return train_loader, color_loader, val_loader, restore_transform
#     return train_loader, val_loader, restore_transform

def createValTestData(datasetname, Dataset, cfg_data,frame_interval,mode ='val'):
    img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*cfg_data.MEAN_STD)
    ])
    if mode == 'val':
        with open(os.path.join( cfg_data.DATA_PATH, cfg_data.VAL_LST), 'r') as txt:
            scene_names = txt.readlines()
            scene_names = [i.strip() for i in scene_names]
        data_loader = []
        for scene_name in scene_names:
            print(scene_name)
            sub_dataset = Dataset(scene_name = scene_name,
                                  base_path=cfg_data.DATA_PATH,
                                  main_transform=None,
                                  img_transform=img_transform,
                                  interval=frame_interval,
                                  target=True,
                                  datasetname = datasetname)
            sub_loader = DataLoader(sub_dataset, batch_size=cfg_data.VAL_BATCH_SIZE,
                                    collate_fn=collate_fn, num_workers=0, pin_memory=True)
            data_loader.append(sub_loader)
        return data_loader
    elif mode == 'test':
        if datasetname=='HT21':
            target = False
            scene_names = ['test/HT21-11', 'test/HT21-12', 'test/HT21-13', 'test/HT21-14', 'test/HT21-15']
        else:
            target =True
            with open(os.path.join( cfg_data.DATA_PATH, cfg_data.TEST_LST), 'r') as txt:
                scene_names = txt.readlines()
                scene_names = [i.strip() for i in scene_names]
        data_loader = []
        for scene_name in scene_names:
            print(scene_name)
            sub_dataset = Dataset(scene_name=scene_name,
                                  base_path=cfg_data.DATA_PATH,
                                  main_transform=None,
                                  img_transform=img_transform,
                                  interval=frame_interval,
                                  target=target,
                                  datasetname=datasetname)
            sub_loader = DataLoader(sub_dataset, batch_size=cfg_data.VAL_BATCH_SIZE,
                                    collate_fn=collate_fn, num_workers=0, pin_memory=True)
            data_loader.append(sub_loader)
        return data_loader


def loading_testset(datasetname, test_interval, mode='test'):

    datasetname = datasetname.upper()
    cfg_data = getattr(setting, datasetname).cfg_data

    Dataset = dataset.TestDataset

    test_loader = createValTestData(datasetname, Dataset, cfg_data,test_interval, mode=mode)

    restore_transform = createRestore(cfg_data.MEAN_STD)
    return test_loader, restore_transform
