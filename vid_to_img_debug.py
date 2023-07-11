from glob import glob
import os
import sys
from tqdm import tqdm

val_res = "save_img_path"
vid_root = "data_path"

def check_dir(dir_path):
    return os.path.isdir(dir_path)

def remove_symbols(vid_name):
    refactered = vid_name
    if "(" in refactered:
        #print(refactered)
        refactered = refactered.replace("(", "\(")
        #print(refactered)
    if ")" in refactered:
        #print(refactered)
        refactered = refactered.replace(")", "\)")
        #print(refactered)
    if "'" in refactered:
        #print(refactered)
        refactered = refactered.replace("'", "")
        #print(refactered)
    if "&" in refactered:
        #print(refactered)
        refactered = refactered.replace("&", "\&")
        #print(refactered)
    if "$" in refactered:
        #print(refactered)
        refactered = refactered.replace("&", "\&")
        #print(refactered)
    if ";" in refactered:
        #print(refactered)
        refactered = refactered.replace(";", "\;")
        #print(refactered)
    return refactered

vid_classes = sorted(glob(vid_root+"/*/"))
print(len(vid_classes))

for class_path in tqdm(vid_classes):
    vid_names = glob(class_path+"/*")
    vid_class = class_path[len(vid_root)+1:-1]
    class_dir = val_res + vid_class

    if not check_dir(class_dir):
        os.mkdir(class_dir)

    for name_path in vid_names:
        vid_name = name_path[len(vid_root)+1+len(vid_class)+1:]
        vid_dir = val_res + vid_class + "/" + vid_name
        without_path = name_path[:len(vid_root)+1+len(vid_class)+1]

        if not check_dir(vid_dir):
            os.mkdir(vid_dir)

        if not os.listdir(vid_dir):
            new_vid_name = remove_symbols(vid_name)
            command = "ffmpeg -i " + without_path + new_vid_name + " <save_img_path>" + vid_class + "/" + new_vid_name +"/%05d.png"
            os.system(command)