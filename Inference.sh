#!/bin/bash

cd src

echo "KAIST pretrained_weights : 30"

CUDA_VISIBLE_DEVICES=0 python inference.py --FDZ original --model-path ../pretrained_weights/KAIST_pretrained_3way_30.pth.tar005 --result-dir save_path

echo "KAIST pretrained_weights : 50"

CUDA_VISIBLE_DEVICES=0 python inference.py --FDZ original --model-path ../pretrained_weights/KAIST_pretrained_50.pth.tar000 --result-dir save_path

echo "KAIST pretrained_weights : 70"

CUDA_VISIBLE_DEVICES=0 python inference.py --FDZ original --model-path ../pretrained_weights/KAIST_pretrained_70.pth.tar000 --result-dir save_path

#echo "LLVIP pretrained_weights : 30"

#CUDA_VISIBLE_DEVICES=0 python inference.py --FDZ original --model-path ../pretrained_weights/LLVIP_pretrained_30.pth.tar000 --result-dir save_path

#echo "LLVIP pretrained_weights : 50"

#CUDA_VISIBLE_DEVICES=0 python inference.py --FDZ original --model-path ../pretrained_weights/LLVIP_pretrained_50.pth.tar000 --result-dir save_path

#echo "LLVIP pretrained_weights : 70"

#CUDA_VISIBLE_DEVICES=0 python inference.py --FDZ original --model-path ../pretrained_weights/LLVIP_pretrained_70.pth.tar000 --result-dir save_path

echo "KAIST final_weights : 30"

CUDA_VISIBLE_DEVICES=0 python inference.py --FDZ original --model-path ../final_weights/KAIST_30.pth.tar000 --result-dir save_path

echo "KAIST final_weights : 50"

CUDA_VISIBLE_DEVICES=0 python inference.py --FDZ original --model-path ../final_weights/KAIST_50.pth.tar000 --result-dir save_path

echo "KAIST final_weights : 70"

CUDA_VISIBLE_DEVICES=0 python inference.py --FDZ original --model-path ../final_weights/KAIST_70.pth.tar000 --result-dir save_path

#echo "LLVIP final_weights : 30"

#CUDA_VISIBLE_DEVICES=0 python inference.py --FDZ original --model-path ../final_weights/LLVIP_30.pth.tar000 --result-dir save_path

#echo "LLVIP final_weights : 50"

#CUDA_VISIBLE_DEVICES=0 python inference.py --FDZ original --model-path ../final_weights/LLVIP_50.pth.tar000 --result-dir save_path

#echo "LLVIP final_weights : 70"

#CUDA_VISIBLE_DEVICES=0 python inference.py --FDZ original --model-path ../final_weights/LLVIP_70.pth.tar000 --result-dir save_path

#CUDA_VISIBLE_DEVICES=0 python inference.py --FDZ original --model-path Model_weight --result-dir save_path #add "--vis"  #if you want to visualize the result