import os
import argparse
from easydict import EasyDict as edict

import torch
import numpy as np

from utils.transforms import *

parser = argparse.ArgumentParser(description='SAMPD Training')
parser.add_argument('--dataset_type', default='KAIST', type=str, help='Dataset type: KAIST or LLVIP')
parser.add_argument('--MP', default=30, type=int, help='Model Percentage')
parser.add_argument('--load_data_setting', default='iterative', type=str, help='Sparse Annotation Type: random or small')

dataset_type = parser.parse_args().dataset_type
MP = parser.parse_args().MP
load_data_setting = parser.parse_args().load_data_setting

SA_type = "small" # sparse annotation type : random or small
set_type = "exist"
teacher =True
pedmixing = True
pretrained = f"../weights/{dataset_type}_pretrained_{MP}.pth.tar000"

# Dataset path
PATH = edict()

if dataset_type == "KAIST":
    PATH.DB_ROOT = '../../../data/kaist-rgbt/'
    PATH.JSON_GT_FILE = os.path.join('kaist_annotations_test20.json' )
elif dataset_type == "LLVIP":
    PATH.DB_ROOT = '../../../data/LLVIP/'
    PATH.JSON_GT_FILE = os.path.join('LLVIP_annotations_test.json' ) 

if MP == 30:
    final_weights = "../weights/KAIST_30.pth.tar000"
elif MP == 50:
    final_weights = "../weights/KAIST_50.pth.tar000"
elif MP == 70:
    final_weights = "../weights/KAIST_70.pth.tar000"

# train
train = edict()


train.student_checkpoint = None
train.teacher_checkpoint = pretrained

train.switching_epoch = 10
train.use_ema = False

train.day = "all"

if dataset_type == "KAIST":
    train.img_set = f"train-all-02.txt"
    if set_type == "exist": train.img_set = f"train-all-02-exist.txt"
elif dataset_type == "LLVIP":
    train.img_set = "LLVIP_train_100p.txt"

train.batch_size = 6 # batch size

train.start_epoch = 0  # start at this epoch
train.epochs = 80  # number of epochs to run without early-stopping
train.epochs_since_improvement = 3  # number of epochs since there was an improvement in the validation metric
train.best_loss = 100.  # assume a high loss at first

train.lr = 1e-3   # learning rate
train.momentum = 0.9  # momentum
train.weight_decay = 5e-4  # weight decay
train.grad_clip = None  # clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32) - you will recognize it by a sorting error in the MuliBox loss calculation
train.teacher = teacher
train.print_freq = 100

train.annotation = "AR-CNN" # AR-CNN, Sanitize, Original 

train.dataset_type = dataset_type
# test & eval
train.MP = MP
test = edict()

test.result_path = '../result'
### coco tool. Save Results(jpg & json) Path

test.day = "all" # all, day, night
if dataset_type == "KAIST":
    test.img_set = f"test-{test.day}-20.txt"
elif dataset_type == "LLVIP":
    test.img_set = f"LLVIP_test_100p.txt" 

test.annotation = "AR-CNN"

test.input_size = [512., 640.]

### test model ~ eval.py
test.checkpoint = "./jobs/best_checkpoint.pth.tar"
test.batch_size = 1

### train_eval.py
test.eval_batch_size = 1


# KAIST Image Mean & STD
## RGB
IMAGE_MEAN = [0.3465,  0.3219,  0.2842]
IMAGE_STD = [0.2358, 0.2265, 0.2274]
## Lwir
LWIR_MEAN = [0.1598]
LWIR_STD = [0.0813]

                    
# dataset
dataset = edict()
dataset.workers = 4
dataset.OBJ_LOAD_CONDITIONS = {    
                                  'train': {'hRng': (12, np.inf), 'xRng':(5, 635), 'yRng':(5, 507), 'wRng':(-np.inf, np.inf)}, 
                                  'test': {'hRng': (-np.inf, np.inf), 'xRng':(5, 635), 'yRng':(5, 507), 'wRng':(-np.inf, np.inf)}, 
                              }


# Fusion Dead Zone
'''
Fusion Dead Zone
The input image of the KAIST dataset is input in order of [RGB, thermal].
Each case is as follows :
orignal, blackout_r, blackout_t, sidesblackout_a, sidesblackout_b, surroundingblackout
'''
FDZ_case = edict()

FDZ_case.original = ["None", "None"]

FDZ_case.blackout_r = ["blackout", "None"]
FDZ_case.blackout_t = ["None", "blackout"]

FDZ_case.sidesblackout_a = ["SidesBlackout_R", "SidesBlackout_L"]
FDZ_case.sidesblackout_b = ["SidesBlackout_L", "SidesBlackout_R"]
FDZ_case.surroundingblackout = ["None", "SurroundingBlackout"]


# main
args = edict(path=PATH,
             train=train,
             test=test,
             dataset=dataset,
             FDZ_case=FDZ_case)

args.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
args.dataset_type = dataset_type

args.exp_time = None
args.exp_name = None

args.n_classes = 3
args.SA_type = SA_type
args.MP = MP
args.pedmixing = pedmixing

args.load_data_setting = load_data_setting

## Semi Unpaired Augmentation
args.upaired_augmentation = ["TT_RandomHorizontalFlip",
                             "TT_FixedHorizontalFlip",
                             "TT_RandomResizedCrop"]

args.want_augmentation = ["RandomHorizontalFlip",
                          "FixedHorizontalFlip",
                          "RandomResizedCrop"]
## Train dataset transform                             
args["train"].img_transform = Compose([ ColorJitter(0.3, 0.3, 0.3), 
                                        ColorJitterLWIR(contrast=0.3)
                                        
                                                                ])
args["train"].co_transform = Compose([
                                        RandomHorizontalFlip(p=0.5), \
                                        RandomResizedCrop([512,640], \
                                                                 scale=(0.25, 4.0), \
                                                                 ratio=(0.8, 1.2)),
                                        ToTensor(), \
                                        Normalize(IMAGE_MEAN, IMAGE_STD, 'R'), \
                                        Normalize(LWIR_MEAN, LWIR_STD, 'T') ], \
                                        args=args)
args["train"].co_transform_weak= Compose([
                                        ToTensor(), \
                                        Normalize(IMAGE_MEAN, IMAGE_STD, 'R'), \
                                        Normalize(LWIR_MEAN, LWIR_STD, 'T') ], \
                                        args=args)

## Test dataset transform
args["test"].img_transform = Compose([ ])   
args["test"].co_transform_weak= Compose([
                                    ToTensor(), \
                                    Normalize(IMAGE_MEAN, IMAGE_STD, 'R'), \
                                    Normalize(LWIR_MEAN, LWIR_STD, 'T') ], \
                                    args=args)
args["test"].co_transform = Compose([Resize(test.input_size), \
                                     ToTensor(), \
                                     Normalize(IMAGE_MEAN, IMAGE_STD, 'R'), \
                                     Normalize(LWIR_MEAN, LWIR_STD, 'T')                        
                                    ])