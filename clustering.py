import matplotlib.pyplot as plt
import os, numpy as np, sys
import shutil
from glob import glob
from sklearn.cluster import KMeans
from sklearn import decomposition
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

trimmed_root = "trimmed_img_root_path"
clustered_root = "clustered_img_root_path"
trimmed_classes = sorted(glob(trimmed_root+"/*/"))
image_h = 64
image_w = 64

def check_dir(dir_path):
    return os.path.isdir(dir_path)

def move_img(from_file_path, to_file_path):
    shutil.copyfile(from_file_path, to_file_path)

def make_dict(key_list, value_list, value_sort = False):
    dict = {}
    for i in range(len(key_list)):
        dict[key_list[i]] = value_list[i]

    if value_sort:
        dict = sorted(dict.items(), key = lambda item: item[1])

    return dict

def k_means_exception_handling(cluster_extracted_img_path, most_large_cluster, cluster_num):
    insufficient_frame_num = cluster_num-len(cluster_extracted_img_path)
    supplement_frames = most_large_cluster[1:1+insufficient_frame_num]
    cluster_extracted_img_path = cluster_extracted_img_path + supplement_frames
    return cluster_extracted_img_path

def extracting(img_paths, clustered_info, cluster_num):
    cluster_extracted_img_path = []
    most_large_cluster = []
    clustered_dict = make_dict(img_paths, clustered_info, value_sort=False)
    for i in range(cluster_num):
        component_list = [k for k, v in clustered_dict.items() if v == i]
        if len(component_list) > len(most_large_cluster):
            most_large_cluster = component_list
        if component_list != []:
            cluster_extracted_img_path.append(component_list[0])

    #print("Before Exception : ", len(cluster_extracted_img_path))
    if len(cluster_extracted_img_path) < 16:
        cluster_extracted_img_path = k_means_exception_handling(cluster_extracted_img_path, most_large_cluster, cluster_num)
    #print("After Exception : ", len(cluster_extracted_img_path))
    return sorted(cluster_extracted_img_path)

def clustering(img_paths, cluster_num, components):
    imgs = []
    for img_path in img_paths:
        img = Image.open(img_path)
        img = img.convert("L")
        img = img.resize((image_h, image_w))
        data = np.asarray(img)
        data = data.flatten()
        imgs.append(data)

    imgs = np.array(imgs)
    pca = decomposition.PCA(n_components=components).fit_transform(imgs)
    print("3. pca clear")

    model = KMeans(init="random", n_clusters=cluster_num, random_state=25)
    model.fit(pca)
    print("4. k-means clear")
    clustered_info = model.labels_
    print(clustered_info)
    extracted_img_paths = extracting(img_paths, clustered_info, cluster_num)
    print("5. extract clear")
    return extracted_img_paths

def copy_img(vid_name_path, clustered_img_dir, extracted_img_paths):
    for img_path in extracted_img_paths:
        only_img = img_path[len(vid_name_path):]
        clustered_img_path = clustered_img_dir + "/" + only_img
        move_img(img_path, clustered_img_path)

for class_path in tqdm(trimmed_classes):
    vid_name_paths = sorted(glob(class_path+"*/"))
    only_class = class_path[len(trimmed_root)+1:-1]
    clustered_class_dir = clustered_root + "/" + only_class

    if not check_dir(clustered_class_dir):
        os.mkdir(clustered_class_dir)
        print("1. clustered_class_dir clear")

    for vid_name_path in vid_name_paths:
        img_paths = sorted(glob(vid_name_path+"*"))
        only_name = vid_name_path[len(class_path):-1]
        clustered_img_dir = clustered_class_dir + "/" + only_name
        #print(img_paths)
        if not check_dir(clustered_img_dir):
            os.mkdir(clustered_img_dir)
            print("2. clustered_img_dir clear")
        else:
            print("** already exist img_dir **")
            continue

        img_num = len(img_paths)
        if img_num > 16:
            extracted_img_paths = clustering(img_paths, cluster_num=16, components=img_num//2)
            print(extracted_img_paths)
            copy_img(vid_name_path, clustered_img_dir, extracted_img_paths)
            print("7. copy_img clear")
        else:
            print(img_paths)
            copy_img(vid_name_path, clustered_img_dir, img_paths)
            print("7. copy_img clear")

