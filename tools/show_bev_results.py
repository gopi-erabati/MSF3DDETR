import cv2
import numpy as np
import os
from tqdm import tqdm
import natsort

data_root = '/mnt/hdd4/achieve-itn/PhD/Code/workdirs/Results/msf3ddetr_v4/images/'

imgdir_names = os.listdir(data_root)
imgdir_names = natsort.natsorted(imgdir_names)

bev_gt_all = []
bev_pred_all = []

for imgdir in tqdm(imgdir_names):
    bev_gt_name = data_root + imgdir + '/bev_gt.png'
    bev_gt = cv2.imread(bev_gt_name)

    bev_pred_name = data_root + imgdir + '/bev_pred.png'
    bev_pred = cv2.imread(bev_pred_name)

    bev_gt_all.append(bev_gt)
    bev_pred_all.append(bev_pred)

for bev_gt, bev_pred in tqdm(zip(bev_gt_all, bev_pred_all)):
    cv2.imshow('bev_gt', bev_gt)
    cv2.waitKey()
    cv2.imshow('bev_pred', bev_pred)
    cv2.waitKey()
