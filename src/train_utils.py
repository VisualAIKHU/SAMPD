import config
import numpy as np
from collections import defaultdict

from model import SSD300, SSD300_3Way

from torchvision.ops import nms, box_iou
import torch.nn.functional as F
import torch
import time
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont
import copy


def init_model():
    args = config.args
    train_conf = config.train
    start_epoch = train_conf.start_epoch
    epochs = train_conf.epochs

    model = SSD300_3Way(n_classes=args.n_classes)
    
    # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
    biases = list()
    not_biases = list()
    for param_name, param in model.named_parameters():
        if param.requires_grad:
            if param_name.endswith('.bias'):
                biases.append(param)
            else:
                not_biases.append(param)
    optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * train_conf.lr},
                                        {'params': not_biases}],
                                lr=train_conf.lr,
                                momentum=train_conf.momentum,
                                weight_decay=train_conf.weight_decay,
                                nesterov=False)

    optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[int(epochs * 0.5), int(epochs * 0.9)],
                                                            gamma=0.1)
    return [model, optimizer, optim_scheduler, start_epoch]

def load_model(checkpoint):
    train_conf = config.train
    epochs = train_conf.epochs
    
    checkpoint = torch.load(checkpoint)
    start_epoch = checkpoint['epoch'] + 1
    model = checkpoint['model']
    optimizer = checkpoint['optimizer']
    optim_scheduler = None
    if optimizer is not None:
        optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(epochs * 0.5)], gamma=0.1)
    return [model, optimizer, optim_scheduler, start_epoch]

def create_teacher_student():
    train_conf = config.train
    student_checkpoint = train_conf.student_checkpoint
    teacher_checkpoint = train_conf.teacher_checkpoint
    
    if student_checkpoint is None:
        student = init_model()
        teacher = load_model(teacher_checkpoint)
        return *student, *teacher
    else:
        student = load_model(student_checkpoint)
        teacher = load_model(teacher_checkpoint)
        return *student, *teacher

def copy_student_to_teacher(teacher_model, student_model):
    for teacher_param, student_param in zip(teacher_model.parameters(), student_model.parameters()):
        teacher_param.data.copy_(student_param.data)

def bayesian_fusion(match_score_vec):
    log_positive_scores = np.log(match_score_vec)
    log_negative_scores = np.log(1 - match_score_vec)
    fused_positive = np.exp(np.sum(log_positive_scores))
    fused_negative = np.exp(np.sum(log_negative_scores))
    fused_positive_normalized = fused_positive / (fused_positive + fused_negative)
    return fused_positive_normalized

def iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    Input boxes are in the format (x, y, w, h).
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
    box1_area = w1 * h1
    box2_area = w2 * h2

    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area

def nms(boxes, scores, iou_threshold):
    indices = np.argsort(scores)[::-1]
    keep_boxes = []

    while len(indices) > 0:
        current = indices[0]
        keep_boxes.append(current)
        if len(indices) == 1:
            break
        current_box = boxes[current]
        rest_boxes = boxes[indices[1:]]
        ious = np.array([iou(current_box, box) for box in rest_boxes])
        indices = indices[1:][ious < iou_threshold]

    return keep_boxes

def detect(detections, len_anno, text="GT"):
    input_size = config.test.input_size
    height, width = input_size
    xyxy_scaler_np = np.array([[width, height, width, height]], dtype=np.float32)
    results = dict()
    boxes = list()
    scores = list()
    labels = list()

    det_boxes_batch, det_labels_batch, det_scores_batch = detections[:3]

    for boxes_t, labels_t, scores_t, image_id in zip(det_boxes_batch, det_labels_batch, det_scores_batch, range(len_anno)):
        boxes_np = boxes_t.cpu().detach().numpy().reshape(-1, 4)
        scores_np = scores_t.cpu().detach().numpy().mean(axis=1).reshape(-1, 1)
        xyxy_np = boxes_np * xyxy_scaler_np
        xywh_np = xyxy_np
        xywh_np[:, 2] -= xywh_np[:, 0]
        xywh_np[:, 3] -= xywh_np[:, 1]
        
        results[image_id + 1] = np.hstack([xywh_np, scores_np])
    
    temp_box = defaultdict(list)
    temp_score = defaultdict(list)
    length = len(results.keys())

    for i in range(length):
        for key, value in results.items():
            if key == i+1:
                for line in results[key]:
                    x, y, w, h, score = line
                    if text == "PL":
                        if float(score) >= 0.5:
                            temp_box[key-1].append([float(x), float(y), float(w), float(h)])
                            temp_score[key-1].append(float(score))
                    else:
                        if float(score) >= 0.5:
                            temp_box[key-1].append([float(x), float(y), float(w), float(h)])
                            temp_score[key-1].append(float(score))

    for num in range(len_anno):
        temp_boxes = temp_box[num]
        score = temp_score[num]
        vis_boxes = np.array(temp_boxes, dtype=np.float64)
        lwir_boxes  = np.array(temp_boxes, dtype=np.float64)
        boxes_vis = [[0,0,0,0,-1,0]]
        boxes_lwir = [[0,0,0,0,-1,0]]

        for i in range(len(vis_boxes)):
            bndbox = [int(i) for i in vis_boxes[i][0:4]]
            bndbox[2] = min( bndbox[2] + bndbox[0], width )
            bndbox[3] = min( bndbox[3] + bndbox[1], height )
            bndbox = [ cur_pt / width if i % 2 == 0 else cur_pt / height for i, cur_pt in enumerate(bndbox) ]
            bndbox.append(1)
            bndbox.append(score[i])
            boxes_vis += [bndbox]

        for i in range(len(lwir_boxes)) :
            bndbox = [int(i) for i in lwir_boxes[i][0:4]]
            bndbox[2] = min( bndbox[2] + bndbox[0], width )
            bndbox[3] = min( bndbox[3] + bndbox[1], height )
            bndbox = [ cur_pt / width if i % 2 == 0 else cur_pt / height for i, cur_pt in enumerate(bndbox) ]
            bndbox.append(1)
            bndbox.append(score[i])
            boxes_lwir += [bndbox]

        boxes_vis = np.array(boxes_vis, dtype=np.float64)
        boxes_lwir = np.array(boxes_lwir, dtype=np.float64)

        boxes_vis = torch.tensor(boxes_vis)
        boxes_lwir = torch.tensor(boxes_lwir)

        if len(boxes_vis.shape) != 1 :
            boxes_vis[1:,4] = 3
        if len(boxes_lwir.shape) != 1 :
            boxes_lwir[1:,4] = 3
        boxes_p = torch.cat((boxes_vis,boxes_lwir), dim=0)
        boxes_p = torch.tensor(list(map(list,set([tuple(bb) for bb in boxes_p.numpy()]))))  
        boxes.append(boxes_p[:,0:4])
        labels.append(boxes_p[:,4])
        scores.append(boxes_p[:,5])
    
    return boxes, scores, labels

def translate_coordinate(box, feature_w, feature_h):
    x1, y1, x2, y2 = box
    x1 = int(x1 * feature_w)
    y1 = int(y1 * feature_h)
    x2 = int(x2 * feature_w)
    y2 = int(y2 * feature_h)

    if x1 == x2:
        if x2 >= feature_w:
            x1 -= 1
        else:
            x2 += 1

    if y1 == y2:
        if y2 >= feature_h:
            y1 -= 1
        else:
            y2 += 1

    return x1, y1, x2, y2

def compute_gap_batch(features, bboxes):
    feature_gaps = []
    
    for bbox in bboxes:
        bbox_gaps = []
        for feature in features:
            x1, y1, x2, y2 = translate_coordinate(bbox, feature.size(2), feature.size(1))
            if x2 - x1 <= 0 or y2 - y1 <= 0:
                continue

            cropped_feature = feature[:, y1:y2, x1:x2]
            if cropped_feature.size(1) > 0 and cropped_feature.size(2) > 0:
                gap = F.avg_pool2d(cropped_feature, kernel_size=cropped_feature.size()[1:]).view(feature.size(0))
                bbox_gaps.append(gap)

        if bbox_gaps:
            feature_gaps.append(torch.mean(torch.stack(bbox_gaps, dim=0), dim=0))

    return feature_gaps

def find_missing_annotations(features, GT_bboxes, PL_bboxes, PL_labels, threshold=0.9):
    device = GT_bboxes[0].device

    exist = False
    miss_bboxes = list()
    miss_labels = list()
    GT_vectors_list = list()
    PL_vectors_list = list()
    pos_PL_list = list()
    neg_PL_list = list()
    for batch_idx in range(features[0].size(0)):
        batch_features = [feature[batch_idx] for feature in features]
        GT_vectors = compute_gap_batch(batch_features, GT_bboxes[batch_idx])
        PL_vectors = compute_gap_batch(batch_features, PL_bboxes[batch_idx])
        GT_vectors_list.append(GT_vectors)
        PL_vectors_list.append(PL_vectors)

        if GT_vectors:
            GT_vectors_tensor = torch.stack(GT_vectors, dim=0)
        else:
            GT_vectors_tensor = None
        miss_bboxes_batch = []
        miss_labels_batch = []
        for PL_idx, PL_vector in enumerate(PL_vectors):
            if GT_vectors_tensor is not None:
                cos_sim = F.cosine_similarity(PL_vector.unsqueeze(0), GT_vectors_tensor, dim=1)
                max_sim = max(cos_sim)

                if max_sim > threshold:
                    miss_bboxes_batch.append(PL_bboxes[batch_idx][PL_idx])
                    miss_labels_batch.append(PL_labels[batch_idx][PL_idx])
                    exist = True
                    pos_PL_list.append(PL_vector)
                elif max_sim < 0.7:
                    neg_PL_list.append(PL_vector)

        miss_bboxes.append(torch.stack(miss_bboxes_batch, dim=0).to(device) if miss_bboxes_batch else torch.tensor([]).to(device))
        miss_labels.append(torch.stack(miss_labels_batch, dim=0).to(device) if miss_labels_batch else torch.tensor([]).to(device))
    
    if exist:
        return GT_vectors_list, PL_vectors_list, pos_PL_list, neg_PL_list, miss_bboxes, miss_labels
    else:
        return None, None, None, None, miss_bboxes, miss_labels

def calc_cos_weight(GT_vectors_list, PL_vectors_list, threshold=0.9):
    cos_weights = list()

    for PL_idx, PL_vectors in enumerate(PL_vectors_list):
        if GT_vectors_list[PL_idx]:
            GT_vectors_tensor = torch.stack(GT_vectors_list[PL_idx], dim=0)
        else:
            GT_vectors_tensor = None
        for PL_idx, PL_vector in enumerate(PL_vectors):
            if GT_vectors_tensor is not None:
                cos_sim = F.cosine_similarity(PL_vector.unsqueeze(0), GT_vectors_tensor, dim=1)
                if max(cos_sim) >= threshold:
                    cos_weights.append(cos_sim[cos_sim >= threshold].mean().item())

    if len(cos_weights) > 0: cos_weight = sum(cos_weights) / len(cos_weights)
    else: cos_weight = 0

    return cos_weight

def calc_contrastive_loss(pos_PL_list, combine_PL_list, tau=0.1):
    device = pos_PL_list[0].device
    
    contrastive_loss = torch.tensor(0.0, requires_grad=True)
    if pos_PL_list:
            pos_PL_vectors = torch.stack(pos_PL_list, dim=0)
    else:
        pos_PL_vectors = None
    if combine_PL_list:
        all_PL_vectors = torch.stack(combine_PL_list, dim=0).to(device)
    else:
        all_PL_vectors = None
    if pos_PL_vectors is not None and all_PL_vectors is not None:
        for pos_PL in pos_PL_list:
            cos_sim_pos = F.cosine_similarity(pos_PL.unsqueeze(0), pos_PL_vectors, dim=1)
            cos_sim_all = F.cosine_similarity(pos_PL.unsqueeze(0), all_PL_vectors, dim=1)

            numerator = torch.sum(torch.exp(cos_sim_pos / tau))
            denominator = torch.sum(torch.exp(cos_sim_all / tau))

            contrastive_loss = contrastive_loss + -torch.log(numerator / denominator) / len(pos_PL_list)

    contrastive_loss = torch.nan_to_num(contrastive_loss, nan=0.0)
    return contrastive_loss

def comb_GT_PL(boxes, labels, pl_bboxes, pl_labels, iou_threshold=0.425):
    comb_boxes, comb_labels = [], []
    cnt = 0
    for batch_idx in range(len(boxes)):
        comb_boxes.append(boxes[batch_idx])
        comb_labels.append(labels[batch_idx])
        if pl_bboxes[batch_idx].nelement() != 0:
            valid_pl_idx = []

            if len(boxes[batch_idx]) == 0:
                comb_boxes[batch_idx] = pl_bboxes[batch_idx]
                comb_labels[batch_idx] = pl_labels[batch_idx]
                continue

            for pl_idx in range(pl_bboxes[batch_idx].shape[0]):
                pl_box = pl_bboxes[batch_idx][pl_idx].unsqueeze(0)
                pl4iou = unnormalize_boxes(pl_box.clone())
                gt4iou = unnormalize_boxes(boxes[batch_idx].clone())
                ious = box_iou(pl4iou, gt4iou)
                if not torch.any(ious > iou_threshold):
                    valid_pl_idx.append(pl_idx)
                    cnt +=1

            valid_pl_bboxes = pl_bboxes[batch_idx][valid_pl_idx]
            valid_pl_labels = pl_labels[batch_idx][valid_pl_idx]

            comb_boxes[batch_idx] = torch.cat((comb_boxes[batch_idx], valid_pl_bboxes), dim=0)
            comb_labels[batch_idx] = torch.cat((comb_labels[batch_idx], valid_pl_labels), dim=0)
        else:
            comb_boxes[batch_idx] = boxes[batch_idx]
            comb_labels[batch_idx] = labels[batch_idx]

    return comb_boxes, comb_labels

def GAP_similarity_based_method(features, pseudo_boxes, pseudo_labels, boxes, labels):
    GT_vectors_list, PL_vectors_list, pos_PL_list, neg_PL_list, miss_bboxes, miss_labels = find_missing_annotations(features, boxes, pseudo_boxes, pseudo_labels)

    if GT_vectors_list is not None and PL_vectors_list is not None:
        cos_weight = calc_cos_weight(GT_vectors_list, PL_vectors_list)
        if len(neg_PL_list) != 0:
            combined_PL_list = pos_PL_list + neg_PL_list
            contrastive_loss = calc_contrastive_loss(pos_PL_list, combined_PL_list)
        else:
            contrastive_loss = torch.tensor(0.0, requires_grad=True).to(boxes[0].device)
        comb_boxes, comb_labels = comb_GT_PL(boxes, labels, miss_bboxes, miss_labels)
        return cos_weight, contrastive_loss, comb_boxes, comb_labels
    else:
        contrastive_loss = torch.tensor(0.0, requires_grad=True).to(boxes[0].device)
        return 0, contrastive_loss, boxes, labels
    
def unnormalize_boxes(boxes, width=640, height=512):
    boxes[:, 0] *= width   # xmin
    boxes[:, 1] *= height  # ymin
    boxes[:, 2] *= width   # xmax
    boxes[:, 3] *= height  # ymax

    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    boxes[:, 2] = w
    boxes[:, 3] = h

    return boxes

def GT_box_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
    box1_area = w1 * h1
    box2_area = w2 * h2

    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area
