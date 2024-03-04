# coding: utf-8
import sys
import argparse
import numpy as np
import cv2
from PIL import Image
from sklearn.decomposition import PCA
import torch
import torch.distributed as dist



def list_to_categorical(label_array, labels):
    label_num = len(labels)
    if label_num == 2:
        file_num = len(label_array)
        ndarray = np.zeros((file_num, label_num), dtype=np.bool)
        for i, label in enumerate(label_array):
            labe = None
            for j in labels:
                if j in label:
                    labe = j
            if labe != None:
                label_index = labels.index(labe)
                ndarray[i, label_index] = 1
    elif label_num > 2:
        file_num = len(label_array)
        label_num = len(labels)
        ndarray = np.zeros((file_num, label_num), dtype=np.bool)
        for i, label in enumerate(label_array):
            labe = None
            for j in labels:
                if j in label:
                    labe = j
            if labe != None:
                label_index = labels.index(labe)
                ndarray[i, label_index] = 1
    else:
        print(labels)
        print("Too few labels")
        sys.exit()
    return ndarray


def crop_center(img, cropx, cropy):
    try:
        y, x, _ = img.shape
    except:
        y, x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy, startx:startx+cropx]

def sizeup_and_center_crop(image_path_list, original_dir):
    for f_n, one_file in enumerate(image_path_list):
        print("\r{} / {}".format(f_n + 1, len(image_path_list)), end="")

        one_file_full_path = original_dir + one_file
        one_img = np.array(Image.open(one_file_full_path))
        try:
            h_n, w_n, ch_n = one_img.shape
        except:
            h_n, w_n = one_img.shape
        if h_n > w_n:
            multiply_rate = h_n / w_n
            new_w_n = 512
            new_h_n = new_w_n * multiply_rate
        else:
            multiply_rate = w_n / h_n
            new_h_n = 512
            new_w_n = new_h_n * multiply_rate

        one_img = cv2.resize(one_img, dsize=(int(new_w_n), int(new_h_n)))
        one_img = crop_center(one_img, 448, 448)

        im = Image.fromarray(one_img)
        im.save(one_file_full_path)


def str2bool(v):
    """
    Converts string to bool type; enables command line 
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_feature_maps(model_family, x, target_class, model):
    if model_family == "convnext":
        features = model.module[0](x)
        features = features.detach()
        class_weights = model.module[1].fc.weight.detach()

    if model_family == "replknet":
        features = model.module.forward_features(x)
        features = features.detach()
        class_weights = model.module.head.weight.detach()

    for i, w in enumerate(class_weights[target_class, :]):
        features[:, i, :, :] *= w

    return features

def get_heatmap_from_features(localization_method, features, model_config, n_comp=30):
    if localization_method == "cam":
        heatmap = torch.mean(features, dim=1).squeeze()
        heatmap = heatmap.cpu()
        heatmap = np.maximum(heatmap, 0)
        heatmap = heatmap / torch.max(heatmap)
        heatmap = heatmap.numpy()

    if localization_method == "pc1":
        fmap_side = model_config["size"]
        features = features.cpu().numpy()
        features = features[0]
        for one_map_n, one_map in enumerate(features):
            fmap_max_v = np.max(one_map)
            fmap_min_v = np.min(one_map)
            one_map_norm = (one_map - fmap_min_v) / (fmap_max_v - fmap_min_v)
            features[one_map_n] = one_map_norm

        features = features.reshape(model_config["unit_n"], fmap_side**2)

        ### get pc1 feature
        pca = PCA(n_components=n_comp)
        pca.fit(features)
        # use_data_latent = pca.transform(features)
        # accumulated_ratio_ = np.add.accumulate(pca.explained_variance_ratio_)
        pc1_matrix = pca.components_[0].reshape(fmap_side, fmap_side)
        pc1_matrix = pc1_matrix / np.max(pc1_matrix)

        ### Decide whether to reverse 0s and 1s.
        mean_pc1_matrix = np.mean(pc1_matrix)
        zero_one_pc1_matrix = np.where(pc1_matrix > mean_pc1_matrix, 1, 0)# Replace pixels that are higher than the average with 1 and pixels that are lower than the average with 0.
        frame_matrix = zero_one_pc1_matrix.copy()
        frame_matrix[1:fmap_side-2, 1:fmap_side-2] = 0
        mean_frame = np.sum(frame_matrix) / ((fmap_side-1)*4)
        if mean_frame > 0.5:# If reversal is required
            pc1_matrix = 1 - pc1_matrix
        heatmap = pc1_matrix

    return heatmap

