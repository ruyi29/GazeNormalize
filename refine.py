import h5py
import cv2
import warp_norm
import matplotlib
import sys
sys.path.append("./FaceAlignment")
import face_alignment
from skimage import io
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from model import gaze_network
from torchvision import transforms
import utils
import pickle
import pandas as pd
import copy
from pt_module import StNet,StRefine
from ipdb import set_trace as st


def Revert_normalization(pred_pitchyaw,RMat):
    from pred_plot import situation_num
    pred_vector = [[] for _ in range(situation_num)]
    for i in range(len(pred_pitchyaw)):
        pred_vector[i] = warp_norm.pitchyaw_to_vector(pred_pitchyaw[i])

    org_pred = [[] for _ in range(situation_num)]
    for i in range(len(pred_vector)):
        for j in range(len(pred_vector[i])):
            # print(RMat[i][j])
            org_pred[i].append(np.dot(np.linalg.inv(RMat[i][j]), pred_vector[i][j].T))
    # print(org_pred[0][0])

    pixel_scale = np.array([0.202, 0.224])

    pred_gc = [[] for _ in range(situation_num)]
    for i in range(len(org_pred)):
        for j in range(len(org_pred[i])):
            pred_gc[i].append(warp_norm.vector_to_gc(org_pred[i][j], pixel_scale, face_center=-600))

    org_tan = np.array([800, 0])  # tan 1600*825

    pred_gc_org = [[] for _ in range(situation_num)]
    for i in range(situation_num):
        pred_gc_org[i] = org_tan + pred_gc[i]

    return pred_gc_org


def PoG_errors(pred, ground_truth):
    from pred_plot import situation_num
    pixel_scale = np.array([0.202, 0.224])
    pred_xerrors=[]
    pred_yerrors=[]
    for i in range(situation_num):
        total_xerrors = 0
        total_yerrors = 0
        for j in range(len(pred[i])):
            total_xerrors=total_xerrors+abs(pred[i][j][0]-ground_truth[i][j][0])
            total_yerrors = total_yerrors + abs(pred[i][j][1] - ground_truth[i][j][1])
        pred_xerrors.append(total_xerrors / (len(pred[i])))
        pred_yerrors.append(total_yerrors / (len(pred[i])))
    pred_xerrors_cm=[]
    pred_yerrors_cm=[]
    for i in range(situation_num):
        pred_xerrors_cm.append(pred_xerrors[i] * 0.1 * pixel_scale[0])
        pred_yerrors_cm.append(pred_yerrors[i] * 0.1 * pixel_scale[1])
    return pred_xerrors_cm, pred_yerrors_cm


def in_screen_percentage(pred):
    from pred_plot import situation_num
    in_screen_net = [[] for _ in range(situation_num)]
    for i in range(situation_num):
        for j in range(len(pred[i])):
            if (0 - 0.05 * 1600 <= pred[i][j][0] <= 1600 + 0.05 * 1600 and 0 - 0.05 * 825 <=
                    pred[i][j][1] <= 825 + 0.05 * 825):
                in_screen_net[i].append(1)
            else:
                in_screen_net[i].append(0)
    # print(in_screen)

    in_screen_percentage = []
    for i in range(situation_num):
        total_in = 0
        for j in range(len(pred[i])):
            total_in = total_in + in_screen_net[i][j]
        in_screen_percentage.append(total_in / len(pred[i]))
    return in_screen_percentage
