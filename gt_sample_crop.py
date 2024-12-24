import cv2
import glob
import os


def is_same_box(rgb_anno, ir_anno):
    rx, ry, rw, rh = map(int, rgb_anno.strip().split(' ')[1:5])
    ix, iy, iw, ih = map(int, ir_anno.strip().split(' ')[1:5])
    if abs(rx - ix) < 5 and abs(ry - iy) < 5 and rw == iw and rh == ih:
        return True
    else:
        return False

remove_file_paths = glob.glob("gt_samples_70/*/*.jpg", recursive=True)
for path in remove_file_paths:
    os.remove(path)

print("remove done")

for rate in [30, 50,70]:
    print(f"start rate {rate}%")
    txt_path = "../src/imageSets/train-all-02-exist.txt"
    base_dir = f'../../data/kaist-rgbt/annotations_paired_miss_small_{rate}'

    txt_format_path = "%s/%s/%s/%s/%s.txt"
    img_format_path = "../../data/kaist-rgbt/images/%s/%s/%s/%s.jpg"
    save_format_path = f"../data/KAIST_gt_samples_{rate}/" + "%s/%s.jpg"
    cnt = 1
    with open(txt_path, 'r', encoding='utf-8') as file:
        total_lines = sum(1 for line in file)
    for txt in open(txt_path):
        if cnt == 1:
            print(f"Progress {cnt} / {total_lines}")
        if cnt % 100 == 0:
           print(f"Progress {cnt} / {total_lines}")
        
        s_id, v_id, i_id = txt.strip().split('/')
        rgb_txt_path = txt_format_path % (base_dir, s_id, v_id, "visible", i_id)
        ir_txt_path = rgb_txt_path.replace("visible", "lwir")
        rgb_img_path = img_format_path % (s_id, v_id, "visible", i_id)
        ir_img_path = rgb_img_path.replace("visible", "lwir")

        rgb_annotations_filtered = []
        ir_annotations_filtered = []
        
        with open(rgb_txt_path, 'r') as file:
            rgb_annotations = file.readlines()[1:] 

        with open(ir_txt_path, 'r') as file:
            ir_annotations = file.readlines()[1:] 

        for ra in rgb_annotations:
            for ia in ir_annotations:
                rgb_annotations_filtered.append(ra)
                ir_annotations_filtered.append(ia)
        
        
        rgb_img = cv2.imread(rgb_img_path)
        ir_img = cv2.imread(ir_img_path)

        for ridx, ra in enumerate(rgb_annotations_filtered):
            x1, y1, w, h = map(int, ra.strip().split(' ')[1:5])
            x2 = x1 + w
            y2 = y1 + h
            rgb_sample = rgb_img[y1:y2, x1:x2, :]
            if rgb_sample is None:
                print("Error: Failed to load image.")
            cv2.imwrite(save_format_path % ("visible", f"{s_id}_{v_id}_{i_id}_{ridx}"), rgb_sample)

        for iidx, ia in enumerate(ir_annotations_filtered):
            x1, y1, w, h = map(int, ia.strip().split(' ')[1:5])
            x2 = x1 + w
            y2 = y1 + h
            ir_sample = ir_img[y1:y2, x1:x2]
            cv2.imwrite(save_format_path % ("lwir", f"{s_id}_{v_id}_{i_id}_{iidx}"), ir_sample)
        cnt +=1

print("all done")