import os
import sys
import shutil
from PIL import Image
from glob import glob
from tqdm import tqdm


trimmed_root = "trimmed_image_root_path"
img_root = "all_img_root_path"
flow_root = "estimated_flow_root_path"
flow_classes = sorted(glob(flow_root+"/*/"))

def check_dir(dir_path):
    return os.path.isdir(dir_path)

def change_value(image):
    i = Image.open(image)
    pixels = i.load()
    width, height = i.size
    all_pixels = []
    pixel_value = 0

    for x in range(width):
        for y in range(height):
            cpixel = pixels[x,y]
            pixel_value += sum(cpixel)
    return int(pixel_value)

def trimming_frames(src_img_dir, dst_img_dir, first_point, second_point):
    print(src_img_dir)
    img_paths = sorted(glob(src_img_dir+"/*"))
    if max(first_point, second_point)-min(first_point, second_point) < 64:
        trimmed_img_paths = img_paths[min(first_point, second_point) + 1:]
    else:
        trimmed_img_paths = img_paths[min(first_point, second_point) + 1:max(first_point, second_point) + 1]
    for from_file_path in trimmed_img_paths:
        only_fime_name = from_file_path[len(src_img_dir)+1:]
        to_file_path = dst_img_dir + "/" + only_fime_name
        shutil.copyfile(from_file_path, to_file_path)


for class_path in tqdm(flow_classes):
    f_names = sorted(glob(class_path+"*/"))
    only_class = class_path[len(flow_root)+1:-1]
    t_class_dir = trimmed_root + only_class
    i_class_dir = img_root + only_class
    if not check_dir(t_class_dir):
        os.mkdir(t_class_dir)

    for f_name in f_names:
        f_path = sorted(glob(f_name+"*"))
        only_name = f_name[len(class_path):-1]
        t_img_dir = trimmed_root + only_class + "/" + only_name
        i_img_dir = img_root + only_class + "/" + only_name

        if not check_dir(t_img_dir):
            os.mkdir(t_img_dir)

        f_path_list = []
        flow_list = []
        if len(f_path) <= 15:
            pass
        else:
            for flow in f_path:
                f_path_list.append(flow)
                f_sum = int(change_value(flow))
                flow_list.append(f_sum)

            flow_list_sort = sorted(flow_list)
            first_point = flow_list.index(flow_list_sort[0])
            second_point = flow_list.index(flow_list_sort[1])

            trimming_frames(i_img_dir, t_img_dir, first_point, second_point)

print("Done!")