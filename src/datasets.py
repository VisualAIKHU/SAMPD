import sys,os,json,time
import glob
import random

import numpy as np
from PIL import Image
import cv2
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance
import copy
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

from utils.utils import *
import shutil

class KAISTPed(data.Dataset):
    """KAIST Detection Dataset Object
    input is image, target is annotation
    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'KAIST')
        condition (string, optional): load condition
            (default: 'Reasonabel')
    """
    def get_new_folder_name(self, base_folder, condition):
        counter = 1
        if condition == "train":
            new_folder = base_folder
            while os.path.exists(new_folder):
                new_folder = f"{base_folder}{counter}"
                counter += 1
        elif condition == "test":
            new_folder = base_folder
            while os.path.exists(new_folder):
                new_folder = f"{base_folder}{counter}"
                counter += 1
            new_folder = f"{base_folder}{counter-2}"
        return new_folder
    
    def __init__(self, args, condition='train'):
        self.args = args
        assert condition in args.dataset.OBJ_LOAD_CONDITIONS
        self.condition = condition
        self.mode = condition
        self.image_set = args[condition].img_set
        self.img_transform = args[condition].img_transform
        self.co_transform = args[condition].co_transform      
        self.co_transform_weak = args[condition].co_transform_weak
        self.cond = args.dataset.OBJ_LOAD_CONDITIONS[condition]
        self.annotation = args[condition].annotation
        self._parser = LoadBox()
        self.load_data_setting = args.load_data_setting
        self.args = args
        self.data = args.dataset_type
        
        if self.data == "KAIST":
            if self.load_data_setting == "iterative" and args.MP != 0:
                src_folder = f'{self.args.path.DB_ROOT}annotations_paired_miss_small_{args.MP}/'
                base_dst_folder = f'{self.args.path.DB_ROOT}annotations_paired_miss_small_{args.MP}_Edit'
                dst_folder_name = self.get_new_folder_name(base_dst_folder, condition)
                dst_folder_name = dst_folder_name
                dst_folder_name = base_dst_folder   
            elif self.load_data_setting == "iterative" and args.MP == 0:
                src_folder = f'{self.args.path.DB_ROOT}annotations_paired/'
                base_dst_folder = f'{self.args.path.DB_ROOT}annotations_paired_Edit'
                dst_folder_name = self.get_new_folder_name(base_dst_folder, condition)
                dst_folder_name = dst_folder_name
                dst_folder_name = base_dst_folder
            else:
                src_folder = f'{self.args.path.DB_ROOT}annotations_paired/'
                base_dst_folder = f'{self.args.path.DB_ROOT}annotations_paired_Edit'
                dst_folder_name = "nothing"
        elif self.data == "LLVIP":
            if self.load_data_setting == "iterative" and args.MP != 0:
                src_folder = f'{self.args.path.DB_ROOT}Annotations_{args.MP}/'
                base_dst_folder = f'{self.args.path.DB_ROOT}Annotations_{args.MP}_Edit'
                dst_folder_name = self.get_new_folder_name(base_dst_folder, condition)
                dst_folder_name = dst_folder_name
                dst_folder_name = base_dst_folder
            elif self.load_data_setting == "iterative" and args.MP == 0:
                src_folder = f'{self.args.path.DB_ROOT}Annotations_Edit/'
                base_dst_folder = f'{self.args.path.DB_ROOT}Annotations_Edit'
                dst_folder_name = self.get_new_folder_name(base_dst_folder, condition)
                dst_folder_name = dst_folder_name
                dst_folder_name = base_dst_folder
            else:
                src_folder = f'{self.args.path.DB_ROOT}Annotations_Edit/'
                base_dst_folder = f'{self.args.path.DB_ROOT}Annotations_Edit'
                dst_folder_name = "nothing"
        

        if condition == "train":
            if self.load_data_setting == "iterative":
                start = time.time()
                print(f'\n\n\nCopied: {src_folder} to {dst_folder_name}\n\n\n')
                print(f"This process will take a few minutes...")
                if os.path.exists(dst_folder_name):
                    print(f"Folder already exists. Removing the folder...")
                    shutil.rmtree(dst_folder_name)
                    print(f"Done!")
                shutil.copytree(src_folder, dst_folder_name)
                end = time.time()
                print(f"Time: {end - start}")
        elif condition == "test":
            print(f"condition : {condition} / dst_folder_name : {dst_folder_name}")
        self.dst_folder = dst_folder_name
        if self.data  == "KAIST":
            self._imgpath = os.path.join('%s', 'images', '%s', '%s', '%s', '%s.jpg')
            self._annopath = os.path.join('%s', 'annotations_paired', '%s', '%s', '%s', '%s.txt')
            if condition == "train":
                self._annopath_exist = os.path.join('%s', 'annotations_paired_miss_small', '%s', '%s', '%s', '%s.txt')
                if self.load_data_setting == "iterative":
                    self._annopath = os.path.join(self.dst_folder, '%s', '%s', '%s', '%s.txt')
                else:
                    self._annopath = os.path.join(src_folder, '%s', '%s', '%s', '%s.txt')
                    self.ids_exist = list()
                    for line in open(os.path.join('./imageSets', "train-all-02-exist.txt")):
                        self.ids_exist.append((self.args.path.DB_ROOT, line.strip().split('/')))

        elif self.data == "LLVIP":
            self._annopath = os.path.join('%s', '%s', '%s.txt')
            self._imgpath = os.path.join('%s', '%s', 'train', '%s.jpg')
            self._testimgpath = os.path.join('%s', '%s', "test", '%s.jpg')  
        
        self.ids = list()
        for line in open(os.path.join('./imageSets', self.image_set)):
            self.ids.append((self.args.path.DB_ROOT, line.strip().split('/')))

        self._brightness_filename = list()
        self._brightness = list()
        
    def __str__(self):
        return self.__class__.__name__ + '_' + self.image_set

    def __getitem__(self, index): 
        vis, lwir,vis2, lwir2, clean_vis, clean_lwir, boxes, labels, boxes2, labels2, return_id,dst_folder = self.pull_item(index)
        return vis, lwir,vis2, lwir2, clean_vis, clean_lwir, boxes, labels, boxes2, labels2, torch.ones(1,dtype=torch.int)*index, return_id, dst_folder
    
    def set_ignore_flags(self, boxes, width, height):
        ignore = torch.zeros( boxes.size(0), dtype=torch.bool)
               
        for ii, box in enumerate(boxes):
                        
            x = box[0] * width
            y = box[1] * height
            w = ( box[2] - box[0] ) * width
            h = ( box[3] - box[1] ) * height

            if  x < self.cond['xRng'][0] or \
                y < self.cond['xRng'][0] or \
                x+w > self.cond['xRng'][1] or \
                y+h > self.cond['xRng'][1] or \
                w < self.cond['wRng'][0] or \
                w > self.cond['wRng'][1] or \
                h < self.cond['hRng'][0] or \
                h > self.cond['hRng'][1]:

                ignore[ii] = 1
        
        boxes[ignore, 4] = -1
        
        labels = boxes[:,4]
        boxes = boxes[:,0:4]

        return boxes, labels
    
    def find_minimum_area(self, saliencyMap, window_size):
        w, h = window_size

        mean = uniform_filter(saliencyMap, size=(h, w))

        min_val = np.min(mean)

        min_pos = np.where(mean == min_val)

        y, x = min_pos[0][0], min_pos[1][0]

        return (x, y), min_val
    
    def calculate_average_rgb_ir(self, rgb_img, ir_img):
        np_rgb_img = np.array(rgb_img)
        np_ir_img = np.array(ir_img)
        if np_rgb_img.shape[0] != np_ir_img.shape[0] or np_rgb_img.shape[1] != np_ir_img.shape[1]:
            np_rgb_img = cv2.resize(np_rgb_img, (np_ir_img.shape[1], np_ir_img.shape[0]))

        if len(np_ir_img.shape) != 3:
            np_ir_img = np.expand_dims(np_ir_img, axis=2)

        combined_img = np.concatenate((np_rgb_img, np_ir_img), axis=2)

        return combined_img.mean()
    
    def find_files(self, starting_with, directory):

        pattern = os.path.join(directory, starting_with + '*')
        files = glob.glob(pattern)
        return files
    
    def load_gt_sample_annotation(self, gt_sample_ids):
        set_id, vid_id, img_id, box_idx = gt_sample_ids
        box_idx = int(box_idx)

        vis_box = None
        lwir_box = None
        for idx, line in enumerate(open(self._annopath % ( self.args.path.DB_ROOT, set_id, vid_id, 'visible', img_id ))) :
            if idx == 0: continue
            if box_idx == idx - 1:
                vis_box = list(map(int, line.strip().split(' ')[1:5]))
        for idx, line in enumerate(open(self._annopath % ( self.args.path.DB_ROOT, set_id, vid_id, 'lwir', img_id))) :
            if idx == 0: continue
            if box_idx == idx - 1:
                lwir_box = list(map(int, line.strip().split(' ')[1:5]))
        
        assert vis_box is not None and lwir_box is not None

        return [vis_box, lwir_box]
        
    
    def select_gt_sample_brightness(self, rgb_img, ir_img):
        brightness = self.calculate_average_rgb_ir(rgb_img, ir_img)
        with open(f"{self.args.dataset_type}_img_rgb_avg_{self.args.MP}_Edit.txt") as f:
            lines = f.readlines()
            lines = [line.strip().split(":") for line in lines] 
        lst = []
        for line in lines:
            bright = abs(float(line[1]) - brightness)
            if bright != 0:
                lst.append((line[0], bright))
        if len(lst) == 0:
            return None, None
        min_bright = min(lst, key=lambda x: x[1])
        rgb_directory = f'{self.args.path.DB_ROOT}{self.args.dataset_type}_gt_samples_{self.args.MP}_Edit/visible'
        if self.args.dataset_type == "KAIST":
            starting_with = min_bright[0].replace("/", "_").split("_")[2]
        else:
            starting_with = min_bright[0]
        rgb_files = self.find_files(starting_with, rgb_directory)
        ir_directory = f'{self.args.path.DB_ROOT}{self.args.dataset_type}_gt_samples_{self.args.MP}_Edit/lwir'

        ir_files = self.find_files(starting_with, ir_directory)
        lst2 = []
        for rgb_file, ir_file in zip(rgb_files, ir_files):
            rgb_img = Image.open(rgb_file)
            ir_img = Image.open(ir_file)
            img_brightness = self.calculate_average_rgb_ir(rgb_img, ir_img)
            difference = abs(img_brightness - brightness)
            lst2.append((rgb_file, ir_file, difference))
        if len(lst2) == 0:
            return None, None
        final_img = min(lst2, key=lambda x: x[1]) 
        final_rgb_img = final_img[0]
        final_lwir_img = final_img[1]

        final_rgb_img = Image.open(final_rgb_img)
        final_lwir_img = Image.open(final_lwir_img).convert("L")

        final_lwir_img = final_lwir_img.resize((final_rgb_img.size[0], final_rgb_img.size[1]))
        return final_rgb_img, final_lwir_img

    def saliency_based_gt_sample_insertion(self, raw_vis_img, raw_lwir_img, vis_gt_sample, lwir_gt_sample):
        raw_vis_img = np.asarray(raw_vis_img).copy()
        raw_lwir_img = np.asarray(raw_lwir_img).copy()
        vis_gt_sample = np.asarray(vis_gt_sample).copy()
        lwir_gt_sample = np.asarray(lwir_gt_sample).copy()

        h, w, _ = vis_gt_sample.shape
        mask = np.ones_like(vis_gt_sample[:, :, 0])
        mask = mask / mask.sum()
        saliency = cv2.saliency.StaticSaliencyFineGrained_create()
        (success, saliencyMap) = saliency.computeSaliency(raw_vis_img)
        saliencyMap = (saliencyMap * 255).astype("uint8")

        result = cv2.filter2D(saliencyMap, -1, mask)
        result = result[139:295+1, :]

        (x, y), min_val = self.find_minimum_area(result, (w, h))
        y = y+139

        mix_val = 0.7

        img_h, img_w = raw_vis_img.shape[:2]
        
        if y + h > img_h:
            y = y - (y + h - img_h) - 1 
        if x + w > img_w:
            x = x - (x + w - img_w) - 1
        if y+h > raw_vis_img.shape[0] or x+w > raw_vis_img.shape[1]:
            if y+h > raw_vis_img.shape[0] and x+w > raw_vis_img.shape[1]:
                origin_vis_val = raw_vis_img[raw_vis_img.shape[0]-h:raw_vis_img.shape[0], raw_vis_img.shape[1]-w:raw_vis_img.shape[1], :]
                origin_lwir_val = raw_lwir_img[raw_lwir_img.shape[0]-h:raw_lwir_img.shape[0], raw_lwir_img.shape[1]-w:raw_lwir_img.shape[1]]
                raw_vis_img[raw_vis_img.shape[0]-h:raw_vis_img.shape[0], raw_vis_img.shape[1]-w:raw_vis_img.shape[1], :] = (origin_vis_val * (1-mix_val) + vis_gt_sample * mix_val).astype(int)
                raw_lwir_img[raw_lwir_img.shape[0]-h:raw_lwir_img.shape[0], raw_lwir_img.shape[1]-w:raw_lwir_img.shape[1]] = (origin_lwir_val * (1-mix_val) + lwir_gt_sample * mix_val).astype(int)

            elif x+w > raw_vis_img.shape[1]:
                origin_vis_val = raw_vis_img[y:y+h, raw_vis_img.shape[1]-w:raw_vis_img.shape[1], :]
                origin_lwir_val = raw_lwir_img[y:y+h, raw_lwir_img.shape[1]-w:raw_lwir_img.shape[1]]
                raw_vis_img[y:y+h, raw_vis_img.shape[1]-w:raw_vis_img.shape[1], :] = (origin_vis_val * (1-mix_val) + vis_gt_sample * mix_val).astype(int)
                raw_lwir_img[y:y+h, raw_lwir_img.shape[1]-w:raw_lwir_img.shape[1]] = (origin_lwir_val * (1-mix_val) + lwir_gt_sample * mix_val).astype(int)
            else:
                origin_vis_val = raw_vis_img[raw_vis_img.shape[0]-h:raw_vis_img.shape[0], x:x+w, :]
                origin_lwir_val = raw_lwir_img[raw_lwir_img.shape[0]-h:raw_lwir_img.shape[0], x:x+w]
                raw_vis_img[raw_vis_img.shape[0]-h:raw_vis_img.shape[0], x:x+w, :] = (origin_vis_val * (1-mix_val) + vis_gt_sample * mix_val).astype(int)
                raw_lwir_img[raw_lwir_img.shape[0]-h:raw_lwir_img.shape[0], x:x+w] = (origin_lwir_val * (1-mix_val) + lwir_gt_sample * mix_val).astype(int)
        else:
            origin_vis_val = raw_vis_img[y:y+h, x:x+w, :]
            origin_lwir_val = raw_lwir_img[y:y+h, x:x+w]
            raw_vis_img[y:y+h, x:x+w, :] = (origin_vis_val * (1-mix_val) + vis_gt_sample * mix_val).astype(int)
            raw_lwir_img[y:y+h, x:x+w] = (origin_lwir_val * (1-mix_val) + lwir_gt_sample * mix_val).astype(int)
        if self.data =="KAIST":
            return Image.fromarray(raw_vis_img), Image.fromarray(raw_lwir_img), ["person", x, y, w, h]
        elif self.data == "LLVIP":
            return Image.fromarray(raw_vis_img), Image.fromarray(raw_lwir_img), [x, y, w, h]

    def pull_item(self, index):
        
        frame_id = self.ids[index]
        if self.data == "KAIST":
            set_id, vid_id, img_id = frame_id[-1]
            return_id = [set_id, vid_id, img_id]
            vis = Image.open( self._imgpath % ( *frame_id[:-1], set_id, vid_id, 'visible', img_id ))
            lwir = Image.open( self._imgpath % ( *frame_id[:-1], set_id, vid_id, 'lwir', img_id ) ).convert('L')
            return_id = [set_id, vid_id, img_id]
            clean_vis = copy.deepcopy(vis)
            clean_lwir = copy.deepcopy(lwir)
            width, height = lwir.size
        elif self.data == "LLVIP":
            img_id = frame_id[-1]
            return_id = [img_id]
            if self.mode == 'train':
                vis = Image.open( self._imgpath % ( *frame_id[:-1], 'visible', img_id[0] )).convert("RGB")
                lwir = Image.open( self._imgpath % ( *frame_id[:-1], 'infrared', img_id[0] ) ).convert('L')
            else: 
                vis = Image.open( self._testimgpath % ( *frame_id[:-1], 'visible', img_id[0] )).convert("RGB")
                lwir = Image.open( self._testimgpath % ( *frame_id[:-1], 'infrared', img_id[0] ) ).convert('L')
            clean_vis = copy.deepcopy(vis)
            clean_lwir = copy.deepcopy(lwir)
            width, height = lwir.size
        
        if self.mode == 'train': 
            vis_boxes = list()
            lwir_boxes = list()
            if self.data == "KAIST":
                if "exist" in self.image_set:
                        for line in open(self._annopath % (set_id, vid_id, 'visible', img_id )) :
                            vis_boxes.append(line.strip().split(' '))
                        for line in open(self._annopath % (set_id, vid_id, 'lwir', img_id)) :
                            lwir_boxes.append(line.strip().split(' '))
                else:
                    if frame_id in self.ids_exist:
                        for line in open(self._annopath_exist % ( *frame_id[:-1], set_id, vid_id, 'visible', img_id )) :
                            vis_boxes.append(line.strip().split(' '))
                        for line in open(self._annopath_exist % ( *frame_id[:-1], set_id, vid_id, 'lwir', img_id)) :
                            lwir_boxes.append(line.strip().split(' '))
                    else:
                        for line in open(self._annopath % ( *frame_id[:-1], set_id, vid_id, 'visible', img_id )) :
                            vis_boxes.append(line.strip().split(' '))
                        for line in open(self._annopath % ( *frame_id[:-1], set_id, vid_id, 'lwir', img_id)) :
                            lwir_boxes.append(line.strip().split(' '))
                vis_boxes = vis_boxes[1:]
                lwir_boxes = lwir_boxes[1:]
            elif self.data == "LLVIP":
                for line in open(self._annopath % ( *frame_id[:-1], f"Annotations_Edit", img_id[0])):
                    vis_boxes.append(line.strip().split())
                for line in open(self._annopath % ( *frame_id[:-1], f"Annotations_Edit", img_id[0])):
                    lwir_boxes.append(line.strip().split())
            boxes_vis = [[0, 0, 0, 0, -1]]
            boxes_lwir = [[0, 0, 0, 0, -1]]
            
            compatibility_index = 0
            if self.data == "KAIST":
                vis_boxes = vis_boxes[1:]
                lwir_boxes = lwir_boxes[1:]
                compatibility_index = 1
            
            if self.args.pedmixing:
                #number_of_mix = random.randint(1, 3)
                number_of_mix = 1

                for i in range(number_of_mix):
                    insertion_box = [0, 0, 0, 0, -1]
                    vis_gt_sample, lwir_gt_sample = self.select_gt_sample_brightness(vis, lwir)
                    if vis_gt_sample is not None and lwir_gt_sample is not None:
                        if len(vis_boxes) != 0 and len(lwir_boxes) != 0:
                            vis_w_lst, vis_h_lst = [], []
                            for i in vis_boxes:
                                if len(i) < 4:
                                    print(f"self._annopath % (set_id, vid_id, 'visible', img_id ) : {self._annopath % (set_id, vid_id, 'visible', img_id )}")
                                    print(f"line : {line}")
                                    print(f"vis_boxes : {vis_boxes}")
                                    print(f"i : {i}")
                                if self.data == "KAIST":
                                    vis_w_lst.append(int(i[3]))
                                    vis_h_lst.append(int(i[4]))
                                else:
                                    vis_w_lst.append(int(i[2]))
                                    vis_h_lst.append(int(i[3]))

                            mean_w, mean_h = sum(vis_w_lst) / len(vis_w_lst), sum(vis_h_lst) / len(vis_h_lst)

                            lwir_w_lst, lwir_h_lst = [], []
                            for i in lwir_boxes:
                                if self.data == "KAIST":
                                    lwir_w_lst.append(int(i[3]))
                                    lwir_h_lst.append(int(i[4]))
                                else:
                                    lwir_w_lst.append(int(i[2]))
                                    lwir_h_lst.append(int(i[3]))

                            if mean_w > 150 or mean_h > 150:
                                mean_w, mean_h = 37, 78
                            vis_gt_sample = vis_gt_sample.resize((int(mean_w), int(mean_h)), resample=Image.BILINEAR)
                            lwir_gt_sample = lwir_gt_sample.resize((int(mean_w), int(mean_h)), resample=Image.BILINEAR)
                            
                            vis, lwir, insertion_box = self.saliency_based_gt_sample_insertion(vis, lwir, vis_gt_sample, lwir_gt_sample)

                            vis_boxes.append(insertion_box)
                            lwir_boxes.append(insertion_box)
                        else:
                            vis_gt_sample = vis_gt_sample.resize((37, 78), resample=Image.BILINEAR)
                            lwir_gt_sample = lwir_gt_sample.resize((37, 78), resample=Image.BILINEAR)
                            
                            vis, lwir, insertion_box = self.saliency_based_gt_sample_insertion(vis, lwir, vis_gt_sample, lwir_gt_sample)

                    vis_boxes.append(insertion_box)
                    lwir_boxes.append(insertion_box)

            boxes_vis = [[0, 0, 0, 0, -1]]
            boxes_lwir = [[0, 0, 0, 0, -1]]

            for i in range(len(vis_boxes)) :
                name = vis_boxes[i][0]
                bndbox = [int(i) for i in vis_boxes[i][0+compatibility_index:4+compatibility_index]]
                bndbox[2] = min( bndbox[2] + bndbox[0], width )
                bndbox[3] = min( bndbox[3] + bndbox[1], height )
                bndbox = [ cur_pt / width if i % 2 == 0 else cur_pt / height for i, cur_pt in enumerate(bndbox) ]
                bndbox.append(1)
                boxes_vis += [bndbox]

            for i in range(len(lwir_boxes)) :
                name = lwir_boxes[i][0]
                bndbox = [int(i) for i in lwir_boxes[i][0+compatibility_index:4+compatibility_index]]
                bndbox[2] = min( bndbox[2] + bndbox[0], width )
                bndbox[3] = min( bndbox[3] + bndbox[1], height )
                bndbox = [ cur_pt / width if i % 2 == 0 else cur_pt / height for i, cur_pt in enumerate(bndbox) ]
                bndbox.append(1)
                boxes_lwir += [bndbox]

            boxes_vis = np.array(boxes_vis, dtype=np.float)
            boxes_lwir = np.array(boxes_lwir, dtype=np.float)

        else :
            boxes_vis = [[0, 0, 0, 0, -1]]
            boxes_lwir = [[0, 0, 0, 0, -1]]
            boxes_vis = np.array(boxes_vis, dtype=np.float)
            boxes_lwir = np.array(boxes_lwir, dtype=np.float)

        if self.img_transform is not None:
            vis, lwir, boxes_vis , boxes_lwir, _ = self.img_transform(vis, lwir, boxes_vis, boxes_lwir)
        if self.co_transform_weak is not None:
            vis2 = copy.deepcopy(vis)
            lwir2 = copy.deepcopy(lwir)
            boxes_vis2 = copy.deepcopy(boxes_vis)
            boxes_lwir2 = copy.deepcopy(boxes_lwir)
            vis2, lwir2, boxes_vis2, boxes_lwir2, _ = self.co_transform_weak(vis2, lwir2, boxes_vis2, boxes_lwir2)
        if self.co_transform is not None:
            pair = 1
            vis, lwir, boxes_vis, boxes_lwir, pair = self.co_transform(vis, lwir, boxes_vis, boxes_lwir, pair)              
            if boxes_vis is None:
                boxes = boxes_lwir
                boxes2 = boxes_lwir2
            elif boxes_lwir is None:
                boxes = boxes_vis
                boxes2 = boxes_vis2
            else : 
                if pair == 1 :
                    if len(boxes_vis.shape) != 1 :
                        boxes_vis[1:,4] = 3
                        boxes_vis2[1:,4] = 3
                    if len(boxes_lwir.shape) != 1 :
                        boxes_lwir[1:,4] = 3
                        boxes_lwir2[1:,4] = 3
                else : 
                    
                    if len(boxes_vis.shape) != 1 :
                        boxes_vis[1:,4] = 1
                        boxes_vis2[1:,4] = 1
                    if len(boxes_lwir.shape) != 1 :
                        boxes_lwir[1:,4] = 2
                        boxes_lwir2[1:,4] = 2
                
                boxes = torch.cat((boxes_vis,boxes_lwir), dim=0)
                boxes = torch.tensor(list(map(list,set([tuple(bb) for bb in boxes.numpy()]))))   
                boxes2 = torch.cat((boxes_vis2,boxes_lwir2), dim=0)
                boxes2 = torch.tensor(list(map(list,set([tuple(bb) for bb in boxes2.numpy()]))))

        boxes, labels = self.set_ignore_flags(boxes, width, height)
        boxes2, labels2 = self.set_ignore_flags(boxes2, width, height)
        return vis, lwir, vis2, lwir2, clean_vis, clean_lwir, boxes, labels, boxes2, labels2, return_id, self.dst_folder

    def __len__(self):
        return len(self.ids)

    def collate_fn(self, batch):
        vis = list()
        lwir = list()
        vis2 = list()
        lwir2 = list()
        clean_vis = list()
        clean_lwir = list()
        boxes = list()
        labels = list()
        boxes2 = list()
        labels2 = list()
        index = list()
        retrun_id = list()
        dst_folder = list()
        

        for b in batch:
            vis.append(b[0])
            lwir.append(b[1])
            vis2.append(b[2])
            lwir2.append(b[3])
            clean_vis.append(b[4])
            clean_lwir.append(b[5])
            boxes.append(b[6])
            labels.append(b[7])
            boxes2.append(b[8])
            labels2.append(b[9])
            index.append(b[10])
            retrun_id.append(b[11])
            dst_folder.append(b[12])

        vis = torch.stack(vis, dim=0)
        lwir = torch.stack(lwir, dim=0)
        vis2 = torch.stack(vis2, dim=0)
        lwir2 = torch.stack(lwir2, dim=0)
  
        return vis, lwir, vis2, lwir2, clean_vis, clean_lwir, boxes, labels, boxes2, labels2, index, retrun_id, dst_folder

class LoadBox(object):
    def __init__(self, bbs_format='xyxy'):
        assert bbs_format in ['xyxy', 'xywh']                
        self.bbs_format = bbs_format
        self.pts = ['x', 'y', 'w', 'h']

    def __call__(self, target, width, height):   
        res = [ [0, 0, 0, 0, -1] ]

        for obj in target.iter('object'):           
            name = obj.find('name').text.lower().strip()            
            bbox = obj.find('bndbox')
            bndbox = [ int(bbox.find(pt).text) for pt in self.pts ]

            if self.bbs_format in ['xyxy']:
                bndbox[2] = min( bndbox[2] + bndbox[0], width )
                bndbox[3] = min( bndbox[3] + bndbox[1], height )

            bndbox = [ cur_pt / width if i % 2 == 0 else cur_pt / height for i, cur_pt in enumerate(bndbox) ]
            
            bndbox.append(1)
            res += [bndbox]
            
        return np.array(res, dtype=np.float)


if __name__ == '__main__':
    from matplotlib import patches
    from matplotlib import pyplot as plt
    from utils.functional import to_pil_image, unnormalize
    import config

    def draw_boxes(axes, boxes, labels, target_label, color):
        for x1, y1, x2, y2 in boxes[labels == target_label]:
            w, h = x2 - x1 + 1, y2 - y1 + 1
            axes[0].add_patch(patches.Rectangle((x1, y1), w, h, fill=False, edgecolor=color, lw=1))
            axes[1].add_patch(patches.Rectangle((x1, y1), w, h, fill=False, edgecolor=color, lw=1))

    args = config.args
    test = config.test

    fig, axes = plt.subplots(1, 2, figsize=(15, 10))

    dataset = KAISTPed(args, condition='test')

    dataset.mode = 'train'

    vis, lwir, boxes, labels, indices = dataset[1300]

    vis_mean = dataset.co_transform.transforms[-2].mean
    vis_std = dataset.co_transform.transforms[-2].std

    lwir_mean = dataset.co_transform.transforms[-1].mean
    lwir_std = dataset.co_transform.transforms[-1].std

    vis_np = np.array(to_pil_image(unnormalize(vis, vis_mean, vis_std)))
    lwir_np = np.array(to_pil_image(unnormalize(lwir, lwir_mean, lwir_std)))

    axes[0].imshow(vis_np)
    axes[1].imshow(lwir_np)
    axes[0].axis('off')
    axes[1].axis('off')

    input_h, input_w = test.input_size
    xyxy_scaler_np = np.array([[input_w, input_h, input_w, input_h]], dtype=np.float32)
    boxes = boxes * xyxy_scaler_np

    draw_boxes(axes, boxes, labels, 3, 'blue')
    draw_boxes(axes, boxes, labels, 1, 'red')
    draw_boxes(axes, boxes, labels, 2, 'green')

    frame_id = dataset.ids[indices.item()]
    set_id, vid_id, img_id = frame_id[-1]
    fig.savefig(f'{set_id}_{vid_id}_{img_id}.png')