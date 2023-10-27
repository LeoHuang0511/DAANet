from typing import Any
from misc.transforms import *

class CutMixShift(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.img_offset_range = cfg.IMG_OFFSET_RANGE
        self.window_offset_range = cfg.WIN_OFFSET_RANGE
        self.crop= RandomCrop( cfg.TRAIN_SIZE)
        self.scale_to_setting = ScaleByRateWithMin(cfg.TRAIN_SIZE[1], cfg.TRAIN_SIZE[0])
        self.rate_range = (0.8,1.2)


    def __call__(self, img1, img2, img3, img4, target1, target2, target3, target4):
        num = 0 
        img1, target1 = self.random_crop_resize(img1, target1)
        num += len(target1['person_id'])
        
        img2, target2 = self.random_crop_resize(img2, target2)
        target2['person_id'] = target2['person_id'] + num
        num += len(target2['person_id'])

        img3, target3 = self.random_crop_resize(img3, target3)
        target3['person_id'] = target3['person_id'] + num
        num += len(target3['person_id'])

        img4, target4 = self.random_crop_resize(img4, target4)
        target4['person_id'] = target4['person_id'] + num


        
        h, w = self.cfg.TRAIN_SIZE

        init_crop = (random.randint(w//4,w*3//4), random.randint(h//4,h*3//4))


        new_crop, window_move = self.window_shift(init_crop)

        shifted_vertices = self.imgs_shift(new_crop, window_move) 


        prev_frame, prev_target = self.cut_mix(img1, img2, img3, img4, target1, target2, target3, target4,(init_crop,init_crop,init_crop,init_crop))
        current_frame, current_target = self.cut_mix(img1, img2, img3, img4, target1, target2, target3, target4, shifted_vertices)
        # current_frame, current_target = self.cut_mix(img1, img2, img3, img4, target1, target2, target3, target4, (new_crop,new_crop,new_crop,new_crop))


        return prev_frame, current_frame, prev_target, current_target

        
    def cut_mix(self, img1, img2, img3, img4, target1, target2, target3, target4, vertices):
        h, w = self.cfg.TRAIN_SIZE

        point1, point2, point3, point4 = vertices

        img1, target1 = self.crop(img1, target1.copy(), point1, (h-point1[1], w-point1[0]))
        img2, target2 = self.crop(img2, target2.copy(), (0, point2[1]), (h-point2[1], point2[0]))
        target2['points'][:,0] += img1.size[0]
        img3, target3 = self.crop(img3, target3.copy(), (point3[0], 0), (point3[1], w-point3[0]))
        target3['points'][:,1] += img1.size[1]

        img4, target4 = self.crop(img4, target4.copy(), (0, 0), (point4[1], point4[0]))
        target4['points'][:,0] += img3.size[0]
        target4['points'][:,1] += img2.size[1]


        assert (img1.size[0]*img1.size[1])+(img2.size[0]*img2.size[1])+(img3.size[0]*img3.size[1])+(img4.size[0]*img4.size[1]) == h*w, "cut size doesn't match !"
        mix_img = Image.new('RGB',(w,h)) 
        mix_img.paste(img1,(0, 0)) 
        mix_img.paste(img2,(img1.size[0], 0)) 
        mix_img.paste(img3,(0, img1.size[1])) 
        mix_img.paste(img4,(img3.size[0], img2.size[1]))
        ids = torch.cat([target1['person_id'],target2['person_id'],target3['person_id'],target4['person_id']], dim=0)

        points = torch.cat([target1['points'],target2['points'],target3['points'],target4['points']], dim=0)

        
        sigma = torch.cat([target1['sigma'],target2['sigma'],target3['sigma'],target4['sigma']], dim=0)


        target = {'person_id':ids, 'points':points, 'sigma':sigma}



        return mix_img, target

        
        
    def imgs_shift(self, new_crop, window_move):
        h, w = self.cfg.TRAIN_SIZE
        img_shift_mode = random.randint(0,1)


        if img_shift_mode == 0: #horizontal mode
            min_offset = max(-new_crop[0],self.img_offset_range[0])
            max_offset = min(w - new_crop[0], self.img_offset_range[1])
            dx1 = random.uniform(min_offset,max_offset)
            dx2 = random.uniform(min_offset,max_offset)
            x1, y1 = new_crop[0] + dx1, new_crop[1]
            x3, y3 = new_crop[0] + dx2, new_crop[1]
            if window_move == False:
                if random.uniform(-1,1) > 0:
                    x1, y1 = new_crop[0], new_crop[1]
                else:
                    x3, y3 = new_crop[0], new_crop[1]
                    
            return (x1,y1), (x1,y1), (x3,y3), (x3,y3)
        
        else: #vertical mode
            min_offset = max(-new_crop[1],self.img_offset_range[0])
            max_offset = min(h - new_crop[1], self.img_offset_range[1])
            dy1 = random.uniform(min_offset,max_offset)
            dy2 = random.uniform(min_offset,max_offset)
            x1, y1 = new_crop[0], new_crop[1] + dy1
            x2, y2 = new_crop[0], new_crop[1] + dy2
            if window_move == False:
                if random.uniform(-1,1) > 0:
                    x1, y1 = new_crop[0], new_crop[1]
                else:
                    x2, y2 = new_crop[0], new_crop[1]
                    
            return (x1,y1), (x2,y2), (x1,y1), (x2,y2)


    def window_shift(self, init_crop):
        h, w = self.cfg.TRAIN_SIZE

        window_move = random.choices([True,False], weights=(4,1))[0]
    
        if window_move == False:
            new_crop = init_crop
        else:
            max_offset = min(self.window_offset_range[1], init_crop[0], init_crop[1], w - init_crop[0], h - init_crop[1])
            min_offset = min(self.window_offset_range[0], init_crop[0], init_crop[1], w - init_crop[0], h - init_crop[1])
            window_offset = random.uniform(min_offset, max_offset)
            phi = random.uniform(0,2)
            # phi = phi +1   # direction needs to + pi (inverse)
            if phi > 1:
                phi = -2 + phi
            # print(phi)
            assert (phi >= -1 or phi <= 1), 'phi is out of range (-1,1) !'
            dx, dy = self.pol2cart(window_offset, phi*np.pi)
            dy*=-1 # y axis of image is up-side down
            new_crop = (init_crop[0]+dx, init_crop[1]+dy)

        return new_crop, window_move


        
    def pol2cart(self, rho, phi):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return(x, y)
    
    def random_crop_resize(self, img, target):
        scale_factor = random.uniform(self.rate_range[0], self.rate_range[1])
        c_h, c_w = int(self.cfg.TRAIN_SIZE[0]/scale_factor), int(self.cfg.TRAIN_SIZE[1]/scale_factor)
        img, target = self.crop(img, target, (0,0),crop_size=(c_h, c_w))
        img, target = self.scale_to_setting(img,target)
        target['person_id'] = torch.arange(0,len(target['points']))

        return img, target


        
