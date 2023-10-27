import os
from PIL import Image
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import xml.etree.ElementTree as ET
import cv2
import glob
import random
import torchvision.transforms.functional as TF
# from .grid_flow_generating import flow_channel2den, flow_channel_select

class ShiftFlowGT:

    def __init__(self, resize_size, grid_size, offset, train=True, deter_shift_phi=None, deter_move=None):
        # self.size = size # W x H, The size that img be resized to before processing
        self.resize_size = resize_size # W x H, The input size of model (default 640 x 360)
        self.grid_size = grid_size
        self.offset = offset #shifting offset of shift pretraining
        self.train = train
        self.deter_shift_phi = deter_shift_phi
        self.deter_move = deter_move
        self.directions = ['UL', 'U', 'UR', 'L', 'O', 'R', 'DL', 'D', 'DR']
        


    def load_frame(self, img_path): 
        # self.labels = {}

        gt_path = img_path.replace('.jpg','.xml').replace('img1','xml')

        tree = ET.parse(gt_path)
        root = tree.getroot()
        img_size = root.find("size")
        img = Image.open(img_path).convert('RGB') #Shift pretrain needs the original size of image to be cropped and shifted
        img_w = int(img_size.find("width").text)
        img_h = int(img_size.find("height").text)


        k = np.zeros((img_h,img_w)) #Shift pretrain needs original size of keypoints map to be cropped and shifted
        key_points = []
        person_ids = []
        # labels = []

        for object in root.iter("object"):
            id = int(object.find("name").text)
            bndbox = object.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            xmax = int(bndbox.find("xmax").text)
            ymin = int(bndbox.find("ymin").text)
            ymax = int(bndbox.find("ymax").text)
            
            y_anno = min(int(((ymax+ymin)/2)), img_h-1)
            x_anno = min(int(((xmax+xmin)/2)), img_w-1)
            k[y_anno,x_anno] += 1
            key_points.append([x_anno,y_anno])
            person_ids.append(id)
        
        target = gaussian_filter(k,3) # In Shift Pretraining, we directly deal with keypoints map insead of target density map
        target = cv2.resize(target,(int(target.shape[1]/self.grid_size),int(target.shape[0]/self.grid_size)),interpolation = cv2.INTER_CUBIC)*(self.grid_size**2)
        patch_index = np.floor(key_points/np.array([self.grid_size,self.grid_size]))
        # self.labels['person_ids'] = np.array(person_ids)
        # self.labels['points'] = np.array(key_points)
        # self.labels['patch_index'] = patch_index.astype('int')
        self.labels['keypoints_map'] = k
        # print(f'{order}, {len(patch_index)}, {patch_index.shape}')
        
        return img, target
    
    def init_crop(self, img, scale):
        self.crop_h = int(img.size[1] * scale)
        self.crop_w = int(img.size[0] * scale)
        self.whole_offset = int(self.offset * (self.crop_h/self.resize_size[1]))
        

        if self.train == True:
            i = random.randint(self.whole_offset,img.size[1]-self.crop_h-self.whole_offset)
            j = random.randint(self.whole_offset,img.size[0]-self.crop_w-self.whole_offset)
        
        elif self.train == False:
            i = (img.size[1]-self.crop_h)//2
            j = (img.size[0]-self.crop_w)//2

        
        
        crop_img = TF.crop(img,i,j,self.crop_h,self.crop_w)
        self.labels['ori_crop_index'] = (i,j)

        return crop_img
    
    def pol2cart(self, rho, phi):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return(x, y)
    
    def shift_img(self, img, index):
        if self.train == True:
            # direction = random.choice(self.directions) # random select a derection
            # d_phi = random.uniform(-0.125,0.125) # random generate the angel in the range of selected direction
            move = random.choices([True,False], weights=(8,1))[0]
            if move==True:
                phi = random.uniform(-1,1)
            elif move==False:
                phi = None

        elif self.train == False and (self.deter_shift_phi != None).all() and (self.deter_move != None).all() and (index != None):
            if self.deter_move[index]==True:
                phi = self.deter_shift_phi[index]
            elif self.deter_move[index]==False:
                phi = None
       
        if phi != None:
            phi = phi +1   # direction needs to + pi (inverse)
            if phi > 1:
                phi = -2 + phi
            print(phi)
            assert (phi >= -1 or phi <= 1), 'phi is out of range (-1,1) !'
            d_x, d_y = self.pol2cart(self.whole_offset, phi*np.pi)
            d_y*=-1 # y axis of image is up-side down
        else:
            d_x = 0
            d_y = 0

        self.labels['c2post_phi'] = phi
        
        i, j = self.labels['ori_crop_index'] # i: h (y), j: w (x)
        post_i = int(min(img.size[1] - self.crop_h -1, max(0, i + d_y))) #constraint not to exceed the orignal image range
        post_j = int(min(img.size[0] - self.crop_w -1, max(0, j + d_x)))

        self.labels['shifted_crop_index'] = (post_i, post_j)
                        
        post_img = TF.crop(img, post_i, post_j, self.crop_h, self.crop_w)

        return post_img
    
    def k2patch_index(self, k): # return a density map in with resize_size

        #size = crop_w, crop_h
        y, x = np.where(k==1)
        key_points = np.concatenate([x[:,None],y[:,None]], axis=1)
        #size = resize size
        rate_h = k.shape[0]/self.resize_size[1]
        rate_w = k.shape[1]/self.resize_size[0]
        key_points[:,1] = (key_points[:,1]/rate_h) # resize y of key_points to resize_size
        key_points[:,0] = (key_points[:,0]/rate_w) # resize x of key_points to resize_size
        key_points[:,1][key_points[:,1] >= self.resize_size[1]] = self.resize_size[1]-1 # index annot exceed resize_size
        key_points[:,0][key_points[:,0] >= self.resize_size[0]] = self.resize_size[0]-1 # index annot exceed resize_size

        patch_index = np.floor(key_points/np.array([self.grid_size, self.grid_size])).astype('int')

      
        return patch_index
    


 

    
    def get_patch_index(self, main_index, shifted_index):
        i, j = main_index
        post_i, post_j = shifted_index
        k = self.labels['keypoints_map']

        main_mask = np.zeros(k.shape) # size = image size
        main_mask[i:i+self.crop_h, j:j+self.crop_w] = 1 # main region (before shifting) mask
        shift_mask = np.zeros(k.shape) # size = image size
        shift_mask[post_i:post_i+self.crop_h, post_j:post_j+self.crop_w] = 1 # main region (before shifting) mask

        boundary_mask = main_mask - shift_mask
        boundary_mask[boundary_mask <= 0] = 0

        self.main_mask = main_mask.copy()
        self.boundary_mask =boundary_mask
        self.shift_mask =shift_mask

        main_mask -= boundary_mask

        main_k = (k*main_mask)[i:(i+self.crop_h), j:(j+self.crop_w)] #crop
        boundary_k = (k*boundary_mask)[i:(i+self.crop_h), j:(j+self.crop_w)] #crop

        main_patch_index = self.k2patch_index(main_k)
        boundary_patch_index = self.k2patch_index(boundary_k)
        
        return main_patch_index, boundary_patch_index
    

        
    def generate_flow_den(self, patch_index_1, patch_index_1_boundary, patch_index_2, patch_index_2_boundary): # input labels of two frames, get the directions of each people, and get the assign channel of people (foward and inverse)
        print('patch:', patch_index_1)
        print('patch2:', patch_index_2)

        directions= patch_index_1 - patch_index_2 # directions order2 -> order1
        directions_inverse= patch_index_2 - patch_index_1 #direction order1 -> order2
        print("dir",directions)
        print("dirinv",directions_inverse)

        flow_channels = flow_channel_select(directions )
        flow_channels_inverse = flow_channel_select(directions_inverse)

        flow_channels = np.concatenate([flow_channels,np.ones(len(patch_index_1_boundary))*9], axis=0) #insert chammel '9' into the index of unmatch people
        flow_channels_inverse = np.concatenate([flow_channels_inverse,np.ones(len(patch_index_2_boundary))*9], axis=0) #insert chammel '9' into the index of unmatch people

        all_patch_index_1 = np.concatenate([patch_index_1, patch_index_1_boundary], axis=0)
        flow_den = flow_channel2den(all_patch_index_1, flow_channels, self.resize_size, self. grid_size)

        all_patch_index_2 = np.concatenate([patch_index_2, patch_index_2_boundary], axis=0)
        flow_den_inverse = flow_channel2den(all_patch_index_2, flow_channels_inverse, self.resize_size, self. grid_size)


        return flow_den, flow_den_inverse


    
    def get_gt(self, img_path, index=None):
        self.labels = {}

        img, _ = self.load_frame(img_path)

        current_img = self.init_crop(img, scale=0.5).resize(self.resize_size)
        post_img = self.shift_img(img, index).resize(self.resize_size)

        init_index = self.labels['ori_crop_index'] # i: h (y), j: w (x)
        shifted_index = self.labels['shifted_crop_index'] # post_i: h (y), post_j: w (x)

        post2c_main_patch_index, post2c_boundary_patch_index = self.get_patch_index(init_index, shifted_index)
        c2post_main_patch_index, c2post_boundary_patch_index = self.get_patch_index(shifted_index, init_index)

     
        post2c_flow_channel, c2post_flow_channel = self.generate_flow_den(post2c_main_patch_index, post2c_boundary_patch_index, c2post_main_patch_index, c2post_boundary_patch_index)



        return current_img, post_img, post2c_flow_channel, c2post_flow_channel


        
def flow_channel_select(directions): # input directions of people moving from a grid to another, output their channel of gt flow density
 
    # device = directions.device
    
    phi = np.arctan2(-directions[:,1], directions[:,0]) / np.pi #Compute the angle of each person from the grid directions, transform to normal polar coordinate (x,-y) -> (x,y)
    phi = phi +1   # direction need to + pi (inverse)
    phi[phi>1] = -2 + phi[phi>1]
    assert ((phi >= -1).all() or (phi <= 1).all()), 'phi is out of range (-1,1) !'

    # (-1,-1) -> UL (0,-1) -> U (1,-1) -> UR (-1,0) -> L (0,0) -> O (0,-1) -> R (-1,1) -> DL (0,1) -> D (1,1) -> DR
    UL_index = np.where((phi >= 0.625) & (phi < 0.875)) #channel 0
    U_index = np.where((phi >= 0.375) & (phi < 0.625)) #channel 1
    RL_index = np.where((phi >= 0.125) & (phi < 0.375)) #channel 2
    L_index = np.where(((phi >= 0.875) & (phi <= 1)) | ((phi >= -1) & (phi < -0.875))) #channel 3
    R_index = np.where(((phi >= 0) & (phi < 0.125)) | ((phi >= -0.125) & (phi <= 0))) #channel 5
    DL_index = np.where((phi >= -0.875) & (phi < -0.625)) #channel 6
    D_index = np.where((phi >= -0.625) & (phi < -0.375)) #channel 7
    DR_index = np.where((phi >= -0.375) & (phi < -0.125)) #channel 8

    channels = np.empty((phi.shape))
    channels[UL_index] = 0
    channels[U_index] = 1
    channels[RL_index] = 2
    channels[L_index] = 3
    channels[R_index] = 5
    channels[DL_index] = 6
    channels[D_index] = 7
    channels[DR_index] = 8
    # print(DR_index)
    
    print('np.sum: ',(directions[:,0] == 0) & (directions[:,1] == 0))
    O_index = np.where((directions[:,0] == 0) & (directions[:,1] == 0)) #channel 4
    # print(torch.sum(directions, dim=1))
    # print(O_index)
    channels[O_index] = 4


    return channels





def flow_channel2den(patch_index, flow_channels, resize_size, grid_size):
    # device = patch_index.device
    flow_dot = np.zeros((10, resize_size[1]//grid_size, resize_size[0]//grid_size))
    # print(patch_index.shape)
    # print(flow_channels.shape)
    # unique_patch, count_patch = torch.unique(torch.insert(patch_index,0,flow_channels,axis=1),axis=0,return_counts=True)
    # print(torch.unsqueeze(flow_channels,1).device)
    # print(patch_index.device)
    # print(flow_channels[:,None])
    # print(flow_channels[:,None].device)
    unique_patch, count_patch = np.unique(np.concatenate((flow_channels[:,None],patch_index),axis=1),axis=0,return_counts=True)
    # print(unique_patch.device)
    # print(count_patch.device)

    print(unique_patch)
    print(count_patch)
    unique_patch = unique_patch.astype('int')
    flow_dot[unique_patch[:,0],unique_patch[:,2],unique_patch[:,1]] = count_patch
    # for i in range(len(flow_density)):
    #     flow_density[i] = gaussian_filter(flow_density[i],3/grid_size)

    return flow_dot
