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
import refine
import os

data_model = 'Gaze360'
situation_num = 2

colors = plt.cm.viridis(np.linspace(0, 1, 4))
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False

model_dir = './models'
state_name = 'spatical_transform_model_fake_eyetracking_dataset_1kg_128_72_error00_valid_random_ii_in_sequence_person_gt_lr_0.1_situation_numsituation_num_full.pt'
state_path = model_dir + '/' + state_name

if situation_num == 9:
    condition_label=[r'白天室内正面光', r'白天室内背面光', r'白天室内侧面光', 
                    r'白天室外任意光', r'晚上室内顶灯光', r'黑暗环境屏幕光',
                    r'黑暗台灯正面光', r'黑暗台灯侧面光', r'黑暗环境外部光',
                    ]
elif situation_num == 2:
    condition_label=[r'upright', r'not_upright']
elif situation_num == 1:
    condition_label=[r'glass']
    # condition_label=[r'noglass']
# print(len(condition_label))


# 得到每张照片的编号用于对应情况然后加入相应列表
def get_folder_names(directory):
    # 获取指定目录下的所有文件和文件夹名称
    all_items = os.listdir(directory)
    folder_names = [int(item) for item in all_items if os.path.isdir(os.path.join(directory, item))]
    folder_names.sort()
    return folder_names

folder_path = './data/pre/test/'
sub_ids = get_folder_names(folder_path)
# sub_ids = [1, 2, 3, 21, 22, 25]
# sub_ids = [16, 36]

file_dict = []
labels = []
R = []

for sub_id in sub_ids:
    image_folder = folder_path + str(sub_id)+'/preprocessed_images'
    csv_file_path = folder_path + str(sub_id)+'/'+'subject' + str(sub_id).zfill(2) + '.csv'
    pkl_file_path = folder_path + str(sub_id) + '/subject' + str(sub_id).zfill(2) + '.pkl'

    df = pd.read_csv(csv_file_path, index_col='image_path')
    image_num = df.shape[0]

    with open(pkl_file_path, 'rb') as fo:
        pkl = pickle.load(fo, encoding='bytes')
        R.extend(pkl)

    for image_path, label, mat_norm, gc_normalized, head_norm, _, _ in df.itertuples(index=True):
        frame_index = int(''.join(filter(str.isdigit, image_path)))
        file_dict.append(frame_index)
        label = list(map(int, label.strip('[]').split()))
        labels.append(label)

# 读取预测后的数据文件
predictions = np.loadtxt('C:/Users/cry/Desktop/GazeEvaluate/Ours_test_predictions/' + data_model + '_predictions.txt', delimiter=',')
predictions = predictions.tolist()

def get_condition_number(file_dict):
    if situation_num == 9:
        return file_dict//264
    elif situation_num == 2:
        return ((file_dict-1)//22)%2
    elif situation_num == 1:
        return 0

ground_truth = [[] for _ in range(situation_num)]
pred = [[] for _ in range(situation_num)]
RMat = [[] for _ in range(situation_num)]

for i in range(len(file_dict)):
    number = get_condition_number(file_dict[i])
    if number >= 0:
        ground_truth[number].append(labels[i])
        pred[number].append(predictions[i])
        RMat[number].append(R[i]['R'])

for i in range(len(ground_truth)):
    ground_truth[i] = np.vstack(ground_truth[i])
    pred[i] = np.vstack(pred[i])

pred_gc_org = refine.Revert_normalization(pred,RMat)
pred_xerrors_cm, pred_yerrors_cm = refine.PoG_errors(pred_gc_org, ground_truth)

print('pred errors:')
print(pred_xerrors_cm)
print(pred_yerrors_cm)

device_level_accuracy = refine.in_screen_percentage(pred_gc_org)
print(device_level_accuracy)

for i in range(situation_num):
    fig = plt.figure(figsize=(10, 12))
    ax = fig.add_subplot()
    rect = plt.Rectangle((0, 0), 1600, 825, edgecolor='r', facecolor='None')
    ax.add_patch(rect)
    for j in range(len(pred[i])):
        plt.scatter(pred_gc_org[i][j][0], pred_gc_org[i][j][1], marker='o', color=colors[3], label=f'Pred')
        plt.scatter(ground_truth[i][j][0], ground_truth[i][j][1], marker='x',color = colors[1], label=f'True')
        plt.arrow(ground_truth[i][j][0], ground_truth[i][j][1], pred_gc_org[i][j][0] - ground_truth[i][j][0],
                  pred_gc_org[i][j][1] - ground_truth[i][j][1], color=colors[3], alpha=0.5)
        if (j == 0):
            plt.legend()

    plt.title(data_model + '_' + condition_label[i] + '_x_vs_y', fontsize=15)
    plt.xlabel('X')
    plt.ylabel('Y')
    # plt.legend()
    plt.tight_layout()
    # plt.show()
    dst_filename = 'test_plot/' + data_model + '_' + condition_label[i] + '_x_vs_y' + '.jpg'
    plt.savefig(dst_filename)
    plt.clf()
