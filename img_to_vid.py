from glob import glob
from tqdm import tqdm
import os
import sys

trimmed_img_root = "trimmed_img_root_path"
trimmed_vid_root = "trimmed_vid_root_path"

trimmed_img_classes = sorted(glob(trimmed_img_root + "*/"))

def check_dir(dir_path):
    return os.path.isdir(dir_path)

for trimmed_img_class in tqdm(trimmed_img_classes):
    trimmed_img_names = sorted(glob(trimmed_img_class + "*/"))
    only_class = trimmed_img_class[len(trimmed_img_root):-1]
    trimmed_vid_class = trimmed_vid_root + only_class

    if not check_dir(trimmed_vid_class):
        os.mkdir(trimmed_vid_class)

    for trimmed_img_name in trimmed_img_names:
        trimmed_img_path = sorted(glob(trimmed_img_name + "/*"))
        only_name = trimmed_img_name[len(trimmed_img_class):-5]
        trimmed_vid_dir = trimmed_vid_class + "/" + only_name

        # if not check_dir(trimmed_vid_dir):
        #     os.mkdir(trimmed_vid_dir)

        if trimmed_img_path == []:
            pass
        else:
            start_number = trimmed_img_path[0][len(trimmed_img_name):len(trimmed_img_name)+5]
            command = "ffmpeg -framerate 30 -pattern_type glob -i '" + trimmed_img_name + '*.png' + "' -c:v libx264 -pix_fmt yuv420p "+ trimmed_vid_class + "/" + only_name + ".mp4"

            os.system(command)


