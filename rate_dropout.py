import numpy as np
import random  

np.random.seed(217)

def print_anno_len(txt_path, base_dir):
    format_path = "%s/%s/%s/%s/%s.txt"
    rgb_total = 0
    ir_total = 0
    for txt in open(txt_path):
        s_id, v_id, i_id = txt.strip().split('/')
        rgb_file_path = format_path % (base_dir, s_id, v_id, "visible", i_id)
        with open(rgb_file_path, 'r') as file:
            rgb_annotations = file.readlines()[1:]
            rgb_total += len(rgb_annotations)
        ir_file_path = rgb_file_path.replace("visible", "lwir")
        with open(ir_file_path, 'r') as file:
            ir_annotations = file.readlines()[1:]
            ir_total += len(ir_annotations)
    print("rgb_total :", rgb_total)
    print("ir_total :", ir_total)
    print("rgb_total + ir_total :", rgb_total + ir_total)

def is_same_box(rgb_anno, ir_anno):
    rx, ry, rw, rh = map(int, rgb_anno.strip().split(' ')[1:5])
    ix, iy, iw, ih = map(int, ir_anno.strip().split(' ')[1:5])
    if abs(rx - ix) < 5 and abs(ry - iy) < 5 and rw == iw and rh == ih:
        return True
    else:
        return False
    
def calculate_area(bbox):
    bbox = list(map(int, bbox.split(' ')[1:5]))
    width = bbox[2]
    height = bbox[3]
    area = width * height
    return area

def reduce_annotations(txt_path, base_dir, removal_count):
    format_path = "%s/%s/%s/%s/%s.txt"
    removed = 0
    ignore_anno_len_flag = True
    while removed < removal_count:
        is_removed = False
        with open(txt_path, "r") as f:
            txts = f.readlines()
        random.shuffle(txts)
        for txt in txts:
            s_id, v_id, i_id = txt.strip().split('/')
            rgb_file_path = format_path % (base_dir, s_id, v_id, "visible", i_id)

            with open(rgb_file_path, 'r') as file:
                rgb_annotations = file.readlines()[1:] 

            ir_file_path = rgb_file_path.replace("visible", "lwir")
            with open(ir_file_path, 'r') as file:
                ir_annotations = file.readlines()[1:]
            
            if ignore_anno_len_flag:
                if len(rgb_annotations) <= 1 and len(ir_annotations) <= 1:
                    continue
            is_removed = True

            print("bf len(rgb_annotations) :", len(rgb_annotations))
            print("bf len(ir_annotations) :", len(ir_annotations))

            comp_anno = None
            if len(rgb_annotations) >= 1 + ignore_anno_len_flag:
                areas = np.array([calculate_area(annotation.strip()) for annotation in rgb_annotations])
                min_idx = areas.argmin()
                comp_anno = rgb_annotations[min_idx]
                del rgb_annotations[min_idx]
                removed += 1
            if len(ir_annotations) >= 1 + ignore_anno_len_flag:
                if comp_anno is not None:
                    for idx, anno in enumerate(ir_annotations):
                        if is_same_box(anno, comp_anno):
                            del ir_annotations[idx]
                            removed += 1
                            break
                else:
                    areas = np.array([calculate_area(annotation.strip()) for annotation in ir_annotations])
                    min_idx = areas.argmin()
                    comp_anno = ir_annotations[min_idx]
                    del ir_annotations[min_idx]
                    removed += 1

            with open(rgb_file_path, 'w') as file:
                file.write("% bbGt version=3\n")
                file.writelines(rgb_annotations)

            with open(ir_file_path, 'w') as file:
                file.write("% bbGt version=3\n")
                file.writelines(ir_annotations)

            print("len(rgb_annotations) :", len(rgb_annotations))
            print("len(ir_annotations) :", len(ir_annotations))
            print()

            if removed >= removal_count: break

        if not is_removed: 
            ignore_anno_len_flag = False

total_box_num = 57297
rate = 30
txt_path = "train-all-02-exist.txt"
base_dir = f'annotations_paired_miss_small_test_{rate}'
removal_count = int(total_box_num * (rate / 100)) 

reduce_annotations(txt_path, base_dir, removal_count)
print_anno_len(txt_path, base_dir)