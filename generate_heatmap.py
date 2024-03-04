import os
import sys
import argparse
import glob
import numpy as np
import pandas as pd
import random
import shutil
import json
import cv2
from scipy import ndimage
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from argparse import ArgumentParser


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim as optim
from torchvision import datasets, transforms
import timm
from sklearn.decomposition import PCA

from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from convnext_func import Net2Head
from replknet_func import *
from utils import *
from dataset_func import *


parser = ArgumentParser('generate heatmap')
parser.add_argument('--model_family', type=str, default='convnext', help='convnext or replknet') 
parser.add_argument('--fine_tuned_weight_name', type=str, default='weights/convnext_base_384_in22ft1k_CUB.pth', help='set your weight file path') 
parser.add_argument('--test_dataset', type=str, default='sample_data', help='sample_data, cub-200-2011, or custom dataset') 
parser.add_argument('--heatmap_output', type=str, default='heatmap_output', help='heatmap output dir name') 
parser.add_argument('--localization_method', type=str, default='cam', help='cam or pc1') 

input_args = parser.parse_args()
model_family = input_args.model_family
fine_tuned_weight_name = input_args.fine_tuned_weight_name
test_dataset = input_args.test_dataset
heatmap_output = input_args.heatmap_output
localization_method = input_args.localization_method

current_path = os.getcwd() + "/"
fine_tuned_weight_fullpath = current_path + fine_tuned_weight_name
test_dir = current_path + "datasets/{}/".format(test_dataset)
heatmap_output_dir = current_path + "{}/".format(heatmap_output)
np_heatmap_output_dir = current_path + "np_{}/".format(heatmap_output)

model_config = {
    "class_n":200,
    "unit_n":1024,
    "input_size": 384, 
    "size":12,
    "lr":1e-05,
    "weight_decay":0.0005
}

### load model
if model_family == 'convnext':
    model_config["model_name"] = "convnext_base_384_in22ft1k"
    loaded_model = timm.create_model(model_config["model_name"], pretrained=False)
    loaded_model = nn.Sequential(*list(loaded_model.children())[:-2])
    loaded_model = nn.Sequential(loaded_model, Net2Head(model_config["class_n"], model_config["unit_n"], model_config["size"]))
elif model_family == 'replknet':
    model_config["model_name"] = "RepLKNet-31B"
    model_config["channels"] = [128,256,512,1024]
    loaded_model = build_model(model_config)
else:
    print('This model is not supported.')

# loaded_model.load_state_dict(torch.load(fine_tuned_weight_fullpath, map_location=torch.device('cpu')))
loaded_model.load_state_dict(torch.load(fine_tuned_weight_fullpath))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
n_gpu = torch.cuda.device_count()
if n_gpu > 0:
    loaded_model = torch.nn.DataParallel(loaded_model, device_ids=[k for k in range(n_gpu)])
loaded_model.cuda()
loaded_model.eval()

### load dataset
img_transform = transforms.Compose([transforms.Resize((model_config["input_size"], model_config["input_size"])), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
dataset_test = datasets.ImageFolder(test_dir, transform=img_transform)
data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False)

### generate heatmap
if os.path.exists(heatmap_output_dir):
    shutil.rmtree(heatmap_output_dir)
os.makedirs(heatmap_output_dir, exist_ok=True)
sample_height = model_config["input_size"]
test_files = data_loader_test.dataset.samples
loop_count = len(test_files)


for img_n, (x, labels) in enumerate(data_loader_test):
    print("\r{} / {}".format(img_n, loop_count), end="")
    x, labels = x.to(device), labels.to(device)

    logit = loaded_model(x)
    h_x = F.softmax(logit, dim=1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.detach().cpu().numpy()
    idx = idx.cpu().numpy()
    target_class = idx[0]

    # get heatmap
    features = get_feature_maps(model_family, x, target_class, loaded_model)
    heatmap = get_heatmap_from_features(localization_method, features, model_config)


    sample_file_name, _ = test_files[img_n]
    sample_file_name_elements = sample_file_name.split("/")

    heatmap = cv2.resize(heatmap, (model_config["input_size"], model_config["input_size"]))
    heatmap = np.uint8(255 * heatmap)
    img = cv2.imread(sample_file_name)
    img = cv2.resize(img, dsize=(model_config["input_size"], model_config["input_size"]))
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap_colored * 0.8 + img
    superimposed_img = np.uint8(255 * superimposed_img / np.max(superimposed_img))
    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    plt.imshow(superimposed_img)
    plt.axis("off")
    plt.savefig(heatmap_output_dir + "heatmap_{}.png".format("0000000"[:-len(str(img_n))] + str(img_n)))
    plt.close()

    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
    class_np_heatmap_output_dir = np_heatmap_output_dir + sample_file_name_elements[-2] + "/"
    if not os.path.exists(class_np_heatmap_output_dir):
        os.makedirs(class_np_heatmap_output_dir, exist_ok=True)
    np.save(class_np_heatmap_output_dir + sample_file_name_elements[-1].split(".")[0], heatmap)



