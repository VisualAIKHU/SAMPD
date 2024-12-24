import os
import time
import copy
import logging
import numpy as np
from typing import Dict

import torch

import config
from model import SSD300_3Way, MultiBoxLoss
from utils import utils
from train_utils import *


def train_epoch(s_model: SSD300_3Way,
                t_model: SSD300_3Way,
                dataloader: torch.utils.data.DataLoader,
                criterion: MultiBoxLoss,
                optimizer: torch.optim.Optimizer,
                logger: logging.Logger,
                epoch: int,
                inference_func,
                **kwargs: Dict) -> float:
    device = next(s_model.parameters()).device
    s_model.train()  # training mode enables dropout
    number = 90
    batch_time = utils.AverageMeter()  # forward prop. + back prop. time
    data_time = utils.AverageMeter()  # data loading time
    losses_sum = utils.AverageMeter()  # loss_sum

    start = time.time()
    if epoch == 0:
        with open (f"./{config.args.dataset_type}_Fully_added_case{number}.txt", "w") as f:
            f.write(f"Start the Record!\n\n")
            f.write(f"epoch: {epoch}\n")
    if epoch > 0:
        with open (f"./{config.args.dataset_type}_Fully_added_case{number}.txt", "a") as f:
            f.write(f"epoch: {epoch}\n")

    added_cnt = 0
    for batch_idx, (image_vis, image_lwir, image_vis2, image_lwir2, clean_vis, clean_lwir, boxes, labels, boxes2, labels2, _, return_id, dst_folder) in enumerate(dataloader):
        data_time.update(time.time() - start)
        crop_vis_img = clean_vis
        crop_lwir_img = clean_lwir

        image_vis = image_vis.to(device)
        image_lwir = image_lwir.to(device)
        image_vis2 = image_vis2.to(device)
        image_lwir2 = image_lwir2.to(device)

        if len(boxes) == 0:
            print(f"len of boxes: {len(boxes)}")
            continue
        if len(boxes) < config.args.train.batch_size:
            print(f"len of boxes: {len(boxes)} is less than {config.args.train.batch_size}")
            continue

        boxes = [box.to(device) for box in boxes]
        labels = [label.to(device) for label in labels]
        boxes2 = [box.to(device) for box in boxes2]
        labels2 = [label.to(device) for label in labels2]

        #image_vis, image_lwir, boxes, labels = CVIU(image_vis, image_lwir, boxes, labels) #Robust Teacher implementation

        # Forward prop.
        predicted_locs_fusion, predicted_scores_fusion, \
        predicted_locs_vis, predicted_scores_vis, \
        predicted_locs_lwir, predicted_scores_lwir, \
        features_fusion, features_vis, features_lwir = s_model(image_vis, image_lwir)  # (N, 8732, 4), (N, 8732, n_classes)

        predicted_locs_fusion2, predicted_scores_fusion2, \
        predicted_locs_vis2, predicted_scores_vis2, \
        predicted_locs_lwir2, predicted_scores_lwir2, \
        features_fusion2, features_vis2, features_lwir2 = s_model(image_vis2, image_lwir2)  # (N, 8732, 4), (N, 8732, n_classes)
        
        with torch.no_grad():
            locs_fusion, classes_scores_fusion, locs_vis, classes_scores_vis, locs_lwir, classes_scores_lwir, _, _, _ = t_model(image_vis, image_lwir)  # (N, 8732, 4), (N, 8732, n_classes)
            locs_fusion2, classes_scores_fusion2, locs_vis2, classes_scores_vis2, locs_lwir2, classes_scores_lwir2, _, _, _ = t_model(image_vis2, image_lwir2)  # (N, 8732, 4), (N, 8732, n_classes)
            f_inference = inference_func(locs_fusion, classes_scores_fusion, min_score=0.1, max_overlap=0.425, top_k=200)
            v_inference = inference_func(locs_vis, classes_scores_vis, min_score=0.1, max_overlap=0.425, top_k=200)
            t_inference = inference_func(locs_lwir, classes_scores_lwir, min_score=0.1, max_overlap=0.425, top_k=200)

            f_inference2 = inference_func(locs_fusion2, classes_scores_fusion2, min_score=0.1, max_overlap=0.425, top_k=200)
            fusion_pseudo_boxes2, fusion_scores2, fusion_pseudo_labels2 = detect(f_inference2, len(boxes2), "PL")
            fusion_pseudo_boxes2 = [pseudo_box.to(device) for pseudo_box in fusion_pseudo_boxes2]
            
        fusion_pseudo_boxes, fusion_scores, fusion_pseudo_labels = detect(f_inference, len(boxes), "PL")
        vis_pseudo_boxes, vis_scores, vis_pseudo_labels = detect(v_inference, len(boxes), "Test")
        lwir_pseudo_boxes, lwir_scores, lwir_pseudo_labels = detect(t_inference, len(boxes), "Test")

        fusion_pseudo_boxes = [pseudo_box.to(device) for pseudo_box in fusion_pseudo_boxes]
        vis_pseudo_boxes = [vis_pseudo_box.to(device) for vis_pseudo_box in vis_pseudo_boxes]
        lwir_pseudo_boxes = [lwir_pseudo_box.to(device) for lwir_pseudo_box in lwir_pseudo_boxes]

        cos_w_fusion, cont_loss_fusion, comb_boxes_fusion, comb_labels_fusion = GAP_similarity_based_method(features_fusion, fusion_pseudo_boxes, fusion_pseudo_labels, boxes, labels)
        cos_w_vis,    cont_loss_vis,    comb_boxes_vis,    comb_labels_vis    = GAP_similarity_based_method(features_vis,    fusion_pseudo_boxes, fusion_pseudo_labels, boxes, labels)
        cos_w_lwir,   cont_loss_lwir,   comb_boxes_lwir,   comb_labels_lwir = GAP_similarity_based_method(features_lwir,   fusion_pseudo_boxes, fusion_pseudo_labels, boxes, labels)
        
        with torch.no_grad():
            cos_w_fusion2, cont_loss_fusion2, comb_boxes_fusion2, comb_labels_fusion2 = GAP_similarity_based_method(features_fusion2, fusion_pseudo_boxes2, fusion_pseudo_labels2, boxes2, labels2)
        if config.args.dataset_type == "KAIST":
            save_format_path = f"{config.args.path.DB_ROOT}KAIST_gt_samples_{config.args.MP}_Edit/" + "%s/%s.jpg"
            save_folder_vis = f"{config.args.path.DB_ROOT}KAIST_gt_samples_{config.args.MP}_Edit/visible"
            save_folder_lwir = f"{config.args.path.DB_ROOT}KAIST_gt_samples_{config.args.MP}_Edit/lwir"
        else:
            save_format_path = f"{config.args.path.DB_ROOT}LLVIP_gt_samples_{config.args.MP}_Edit/" + "%s/%s.jpg"
            save_folder_vis = f"{config.args.path.DB_ROOT}LLVIP_gt_samples_{config.args.MP}_Edit/visible"
            save_folder_lwir = f"{config.args.path.DB_ROOT}LLVIP_gt_samples_{config.args.MP}_Edit/lwir"

        if config.args.load_data_setting == "iterative":
            for i in range(len(fusion_pseudo_boxes2)):

                if len(fusion_scores2[i]) !=0:
                    for j in range(len(fusion_scores2[i])):
                        iou_with_original_box = list()

                        if fusion_scores2[i][j] > float(f"0.{number}") and cos_w_fusion2 > 0.9 and fusion_pseudo_labels2[i][j] == 3:
                            x, y, w, h = fusion_pseudo_boxes2[i][j]
                            x, y, w, h = int(x*640) , int(y*512), int(w*640), int(h*512)
                            
                            w, h = w-x, h-y

                            x2 = x + w
                            y2 = y + h
                            if config.args.dataset_type == "KAIST":
                                save_path_vis = os.path.join(dst_folder[i], return_id[i][0], return_id[i][1], "visible", f"{return_id[i][2]}.txt")
                                save_path_lwir = os.path.join(dst_folder[i], return_id[i][0], return_id[i][1], "lwir", f"{return_id[i][2]}.txt")
                            else:
                                save_path_vis = os.path.join(dst_folder[i], f"{return_id[i][0][0]}.txt")
                                save_path_lwir = os.path.join(dst_folder[i], f"{return_id[i][0][0]}.txt")

                            box = unnormalize_boxes(copy.deepcopy(boxes2[i]))
                            
                            label = labels2[i]
                            for lines, l in zip(box,label):
                                if l == 3:
                                    lines = [int(b) for b in lines]
                                    if len(lines) > 0:
                                        original_box = lines
                                        iou_with_original_box.append(GT_box_iou(original_box, [x, y, w, h]))

                            
                            if len(iou_with_original_box) > 0:
                                    if all(iou < 0.425 for iou in iou_with_original_box):
                                        if config.args.dataset_type == "KAIST":
                                            with open(save_path_vis, "a") as f:
                                                f.write(f"person {x} {y} {w} {h} 0 0 0 0 0 0 0\n")
                                                f.flush()
                                        else:
                                            with open(save_path_vis, "a") as f:
                                                f.write(f"{x} {y} {w} {h}\n")
                                                f.flush()
                                        try:
                                            rgb_sample_np = np.array(crop_vis_img[i])
                                            rgb_sample = rgb_sample_np[y:y2, x:x2, :]
                                            rgb_sample = Image.fromarray(rgb_sample.astype(np.uint8))
                                        except ValueError as e:
                                            continue
                                        
                                        if config.args.dataset_type == "KAIST":
                                            base_filename = f"{return_id[i][2]}_"
                                        else:
                                            base_filename = f"{return_id[i][0][0]}_"
                                        existing_files = [f for f in os.listdir(save_folder_vis) if f.startswith(base_filename)]
                                        existing_ridxs = [int(f.split('_')[-1].split('.')[0]) for f in existing_files if f.split('_')[-1].split('.')[0].isdigit()]
                                        ridx = max(existing_ridxs) + 1 if existing_ridxs else 0


                                        if config.args.dataset_type == "KAIST":
                                            save_path = save_format_path % ("visible", f"{return_id[i][2]}_{ridx}")
                                        else:
                                            save_path = save_format_path % ("visible", f"{return_id[i][0][0]}_{ridx}")

                                        rgb_sample.save(save_path)


                                        
                                        if config.args.dataset_type == "KAIST":
                                            with open(save_path_lwir, "a") as f:
                                                f.write(f"person {x} {y} {w} {h} 0 0 0 0 0 0 0\n")
                                                f.flush()

                                        try:
                                            lwir_sample_np = np.array(crop_lwir_img[i])

                                            lwir_sample = lwir_sample_np[y:y2, x:x2]

                                            lwir_sample = Image.fromarray(lwir_sample.astype(np.uint8))
                                        except ValueError as e:
                                            continue                        
                                        existing_files = [f for f in os.listdir(save_folder_lwir) if f.startswith(base_filename)]
                                        existing_ridxs = [int(f.split('_')[-1].split('.')[0]) for f in existing_files if f.split('_')[-1].split('.')[0].isdigit()]
                                        ridx = max(existing_ridxs) + 1 if existing_ridxs else 0
                                        if config.args.dataset_type == "KAIST":
                                            save_path = save_format_path % ("lwir", f"{return_id[i][2]}_{ridx}")
                                        else:
                                            save_path = save_format_path % ("lwir", f"{return_id[i][0][0]}_{ridx}")
                                        lwir_sample.save(save_path)
                                        lwir_sample_np_expanded = np.expand_dims(lwir_sample_np, axis=2)
                                        combined_img = np.concatenate((rgb_sample_np, lwir_sample_np_expanded), axis=2)
    
                                        if ridx == 0:
                                            lwir_sample_np_expanded = np.expand_dims(lwir_sample_np, axis=2)
                                            combined_img = np.concatenate((rgb_sample_np, lwir_sample_np_expanded), axis=2)
                                            if config.args.dataset_type == "KAIST":
                                                with open(f"./{config.args.dataset_type}_img_rgb_avg_{config.args.MP}_Edit.txt", "a") as f:
                                                    
                                                    f.write(f"{return_id[i][0]}/{return_id[i][1]}/{return_id[i][2]}:{combined_img.mean()}\n")
                                                    f.flush()
                                                    
                                            else:
                                                with open(f"./{config.args.dataset_type}_img_rgb_avg_{config.args.MP}_Edit.txt", "a") as f:
                                                    f.write(f"{return_id[i][0][0]}:{combined_img.mean()}\n")
                                                    f.flush()
                                                    
                                        added_cnt += 1
                                        with open (f"./{config.args.dataset_type}_Fully_added_case{number}.txt", "a") as f:
                                            if config.args.dataset_type == "KAIST":
                                                f.write(f"save_path_lwir: {save_path_lwir} / predicted score : {fusion_scores2[i][j]} / person {x} {y} {w} {h} 0 0 0 0 0 0 0 / added_cnt: {added_cnt}\n")
                                                f.flush()
                                            elif config.args.dataset_type == "LLVIP":
                                                f.write(f"save_path_lwir: {save_path_lwir} / predicted score : {fusion_scores2[i][j]} / {x} {y} {w} {h} / added_cnt: {added_cnt}\n")
                                                f.flush()
                            elif len(iou_with_original_box) == 0 and len(box) == 0:
                                if config.args.dataset_type == "KAIST":
                                    with open(save_path_vis, "a") as f:
                                        f.write(f"person {x} {y} {w} {h} 0 0 0 0 0 0 0\n")
                                        f.flush()
                                else:
                                    with open(save_path_vis, "a") as f:
                                        f.write(f"{x} {y} {w} {h}\n")
                                        f.flush()
                                try:
                                    rgb_sample_np = np.array(crop_vis_img[i])
                                    rgb_sample = rgb_sample_np[y:y2, x:x2, :]
                                    rgb_sample = Image.fromarray(rgb_sample.astype(np.uint8))
                                except ValueError as e:             

                                    continue                        
                                        
                                if config.args.dataset_type == "KAIST":
                                    base_filename = f"{return_id[i][2]}_"
                                else:
                                    base_filename = f"{return_id[i][0][0]}_"
                                existing_files = [f for f in os.listdir(save_folder_vis) if f.startswith(base_filename)]
                                existing_ridxs = [int(f.split('_')[-1].split('.')[0]) for f in existing_files if f.split('_')[-1].split('.')[0].isdigit()]
                                ridx = max(existing_ridxs) + 1 if existing_ridxs else 0
                                
                                

                                if config.args.dataset_type == "KAIST":
                                    save_path = save_format_path % ("visible", f"{return_id[i][2]}_{ridx}")
                                else:
                                    save_path = save_format_path % ("visible", f"{return_id[i][0][0]}_{ridx}")
                                rgb_sample.save(save_path)
                                
                                if config.args.dataset_type == "KAIST":
                                    with open(save_path_lwir, "a") as f:
                                        f.write(f"person {x} {y} {w} {h} 0 0 0 0 0 0 0\n")
                                        f.flush()

                                try:
                                    lwir_sample_np = np.array(crop_lwir_img[i])
                                    lwir_sample = lwir_sample_np[y:y2, x:x2]
                                    lwir_sample = Image.fromarray(lwir_sample.astype(np.uint8))
                                except ValueError as e: 
                                    continue
                                    

                                if config.args.dataset_type == "KAIST":
                                    base_filename = f"{return_id[i][2]}_"
                                else:
                                    base_filename = f"{return_id[i][0][0]}_"
                                existing_files = [f for f in os.listdir("path_to_save_folder") if f.startswith(base_filename)]
                                
                                existing_ridxs = [int(f.split('_')[-1].split('.')[0]) for f in existing_files if f.split('_')[-1].split('.')[0].isdigit()]
                                

                                ridx = max(existing_ridxs) + 1 if existing_ridxs else 0
                                
                                if config.args.dataset_type == "KAIST":
                                    save_path = save_format_path % ("lwir", f"{return_id[i][2]}_{ridx}")
                                else:
                                    save_path = save_format_path % ("lwir", f"{return_id[i][0][0]}_{ridx}")
                                lwir_sample.save(save_path)
                                added_cnt += 1
                                with open (f"./{config.args.dataset_type}_Fully_added_case{number}.txt", "a") as f:
                                    if config.args.dataset_type == "KAIST":
                                        f.write(f"save_path_lwir: {save_path_lwir} / predicted score : {fusion_scores2[i][j]} / person {x} {y} {w} {h} 0 0 0 0 0 0 0 / added_cnt: {added_cnt}\n")
                                        f.flush()
                                    elif config.args.dataset_type == "LLVIP":
                                        f.write(f"save_path_lwir: {save_path_lwir} / predicted score : {fusion_scores2[i][j]} / {x} {y} {w} {h} / added_cnt: {added_cnt}\n")
                                        f.flush()
                                if ridx == 0:
                                    lwir_sample_np_expanded = np.expand_dims(lwir_sample_np, axis=2)
                                    combined_img = np.concatenate((rgb_sample_np, lwir_sample_np_expanded), axis=2)
                                    if config.args.dataset_type == "KAIST":
                                        with open(f"./{config.args.dataset_type}_img_rgb_avg_{config.args.MP}_Edit.txt", "a") as f:
                                            f.write(f"{return_id[i][0]}/{return_id[i][1]}/{return_id[i][2]}:{combined_img.mean()}\n")
                                            f.flush()
                                            
                                    else:
                                        with open(f"./{config.args.dataset_type}_img_rgb_avg_{config.args.MP}_Edit.txt", "a") as f:
                                            f.write(f"{return_id[i][0][0]}:{combined_img.mean()}\n")
                                            f.flush()
        
        
        loss_fusion, _, _, n_positives = criterion(predicted_locs_fusion, predicted_scores_fusion, comb_boxes_fusion, comb_labels_fusion)
        loss_vis,    _, _,           _ = criterion(predicted_locs_vis,    predicted_scores_vis,    comb_boxes_vis,    comb_labels_vis)
        loss_lwir,   _, _,           _ = criterion(predicted_locs_lwir,   predicted_scores_lwir,   comb_boxes_lwir,   comb_labels_lwir)
        loss = cos_w_fusion * loss_fusion + cos_w_vis * loss_vis + cos_w_lwir * loss_lwir  + cont_loss_fusion + cont_loss_vis + cont_loss_lwir
        if torch.isnan(loss).any(): continue\
        

        optimizer.zero_grad()
        loss.backward()

        if kwargs.get('grad_clip', None):
            utils.clip_gradient(optimizer, kwargs['grad_clip'])


        optimizer.step()

        losses_sum.update(loss.item())
        batch_time.update(time.time() - start)

        start = time.time()

        if batch_idx % kwargs.get('print_freq', 10) == 0:
            logger.info('Iteration: [{0}/{1}]\t'
                        'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'num of Positive {Positive}\t'.format(batch_idx, len(dataloader),
                                                              batch_time=batch_time,
                                                              data_time=data_time,
                                                              loss=losses_sum,
                                                              Positive=n_positives))
    return losses_sum.avg
