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

colors = plt.cm.viridis(np.linspace(0, 1, 4))
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False
model_dir = './models'
state_name = 'spatical_transform_model_fake_eyetracking_dataset_1kg_128_72_error00_valid_random_ii_in_sequence_person_gt_lr_0.1_situation_numsituation_num_full.pt'
state_path = model_dir + '/' + state_name

data_model = 'Gaze360'             # 使用的模型（'Columbia', 'EVE', 'Gaze360', 'MPII', 'Ours', 'xgaze')
situation_num = 2                  # 可视化情况的数量
folder_path = './data/pre/test/'   # 数据的存储文件

if situation_num == 9:     # 光照情况
    condition_label=['白天室内正面光', '白天室内背面光', '白天室内侧面光', 
                     '白天室外任意光', '晚上室内顶灯光', '黑暗环境屏幕光',
                     '黑暗台灯正面光', '黑暗台灯侧面光', '黑暗环境外部光',
                    ]
elif situation_num == 2:   # 头部姿态
    condition_label=['upright', 'not_upright']
elif situation_num == 1:   # 配戴眼镜
    condition_label=['glass']
    # condition_label=[r'noglass']

# 得到每张照片的编号用于对应情况然后加入相应列表
def get_folder_names(directory):
    # 获取指定目录下的所有文件和文件夹名称
    all_items = os.listdir(directory)
    folder_names = [int(item) for item in all_items if os.path.isdir(os.path.join(directory, item))]
    folder_names.sort()
    return folder_names

# 获取需要遍历的主体编号
if condition_label == ['glass']:
    sub_ids = [1, 2, 3, 21, 22, 25]
elif condition_label == ['noglass']:
    sub_ids = [16, 36]
else:
    sub_ids = get_folder_names(folder_path)



##TODO：我建议要么用csv，要么用pkl，这样真的好蠢啊

frame_index = []   # 照片的编号
labels = []      # 视线在屏幕上的真值
R = []           # 包含R的pkl文件

for sub_id in sub_ids:
    image_folder = folder_path + str(sub_id)+'/preprocessed_images'
    csv_file_path = folder_path + str(sub_id)+'/'+'subject' + str(sub_id).zfill(2) + '.csv'
    pkl_file_path = folder_path + str(sub_id) + '/subject' + str(sub_id).zfill(2) + '.pkl'

    df = pd.read_csv(csv_file_path, index_col='image_path')
    image_num = df.shape[0]

    with open(pkl_file_path, 'rb') as fo:
        pkl = pickle.load(fo, encoding='bytes')
        R.extend(pkl)

    for image_path, label, mat_norm, _, _, _, _ in df.itertuples(index=True):
        index = int(''.join(filter(str.isdigit, image_path)))
        frame_index.append(index)
        label = list(map(int, label.strip('[]').split()))
        labels.append(label)

# 读取预测后的数据文件
name = ''  # 用于区分是否佩戴眼镜（补充文件名）
if condition_label == ['glass']:
    name = '_glass'
elif condition_label == ['noglass']:
    name = '_noglass'
predictions = np.loadtxt('C:/Users/cry/Desktop/GazeEvaluate/Ours_test_predictions/' + data_model + name + '_predictions.txt', delimiter=',')
predictions = predictions.tolist()

# 通过照片的编号，获取对应情况的序号
def get_condition_number(frame_index):
    if situation_num == 9:
        return frame_index // 264
    elif situation_num == 2:
        return ((frame_index - 1) // 22) % 2
    elif situation_num == 1:
        return 0

# 将对应情况的数据分类存储
ground_truth = [[] for _ in range(situation_num)]   # 视线在屏幕上的真值
pred = [[] for _ in range(situation_num)]           # 预测视线
RMat = [[] for _ in range(situation_num)]           # 旋转矩阵

for i in range(len(frame_index)):
    number = get_condition_number(frame_index[i])
    if number >= 0:
        ground_truth[number].append(labels[i])
        pred[number].append(predictions[i])
        RMat[number].append(R[i]['R'])

for i in range(len(ground_truth)):
    ground_truth[i] = np.vstack(ground_truth[i])
    pred[i] = np.vstack(pred[i])


# 计算误差
pred_gc_org = refine.Revert_normalization(pred,RMat)
pred_xerrors_cm, pred_yerrors_cm = refine.PoG_errors(pred_gc_org, ground_truth)
device_level_accuracy = refine.in_screen_percentage(pred_gc_org)
print('pred errors:')
print(pred_xerrors_cm)
print(pred_yerrors_cm)
print(device_level_accuracy)


# 可视化绘图 & 存储
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
