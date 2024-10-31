import os
import cv2
import csv
import numpy as np
import pandas as pd
from PIL import Image
import warp_norm
import pickle
import h5py

sub_id = 36
index = 0

cam = './Calibration/cam.xml'  # this is camera calibration information file obtained with OpenCV
fs = cv2.FileStorage(cam, cv2.FILE_STORAGE_READ)
w = 1600
h = 825
pixel_scale = 0.211667
camera_matrix = fs.getNode('Camera_Matrix').mat() # camera calibration information is used for data normalization
camera_distortion = fs.getNode('Distortion_Coefficients').mat()

# 图像文件所在的文件夹路径
image_folder_path = './data/raw/test/'+str(sub_id)+'/Photo'

# 预处理后的数据存储路径
save_dir = './data/pre/test/'+str(sub_id)+'/preprocessed_images'
os.makedirs(save_dir, exist_ok=True)

# 数据集列表
dataset = []

# 标签列表
load_labels = []
with open(os.path.join('./data/raw/test/'+str(sub_id), 'coordinate.txt'), 'r') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        # print(row)
        load_labels.append(row)
# print(load_labels)
gaze_centers =[[int(i[-3]), int(i[-2])] for i in load_labels[0:] if [int(i[-3]), int(i[-2])] != [-1, -1]]
# print(gaze_centers)
# print(gaze_centers.__len__())
# exit()

model1, model2, model3 = warp_norm.xmodel()
# 遍历图像文件夹
for filename in sorted(os.listdir(image_folder_path), key=lambda x: int(os.path.splitext(x)[0])):
    if filename.endswith(".jpg"):
        # 判断是否为注视屏幕内的图片
        num = int(os.path.splitext(filename)[0])
        if num % 22 == 21 or num % 22 == 0:
            continue
        # 构建图像文件的完整路径
        image_path = os.path.join(image_folder_path, filename)
        print('image_path', image_path)
        # 读取标签
        label = np.array(gaze_centers[index])
        index += 1
        # print(label)
        # exit()

        # 读取图像
        image = cv2.imread(image_path)
        # print(h)
        image, gaze_center, R, Ear, face_center_in_img, hrn = warp_norm.GazeNormalization(image, camera_matrix,camera_distortion,label,w,h,predictor=model1, face_detector=model2, eve_detector=model3)
        # print("begin:", gaze_center, R, Ear, face_center_in_img, hrn)
        # exit()
        if(Ear == -1):
            continue
        # 保存预处理后的图像
        scale=np.array([[1,1,1],[1,1,1],[0.8,0.8,0.8]])
        R = R * scale
        # print(R)
        # 保存预处理后的图像
        save_path = os.path.join(save_dir, f'preprocessed_image_{filename}')
        cv2.imwrite(save_path, image)
        print('face_center_in_img', face_center_in_img)
        # 添加到数据集列表
        # face_area_label = test.get_face_center_label(face_center_in_img[0], face_center_in_img[1], (320, 240))
        # print(face_area_label)
        # dataset.append({'frame_index': f'preprocessed_image_{filename}', 'cam_index': 0, 'face_mat_norm': R, 'face_head_pose':hrn, 'face_gaze':gaze_center})
        dataset.append({'image_path': f'preprocessed_image_{filename}', 'original_label': label, 'R': R, 'gc_normalized': gaze_center})
        # dataset.append({'image_path': f'preprocessed_image_{filename}', 'original_label': label, 'R': R, 'face_area_label':face_area_label})


pickle_file_path = os.path.join('./data/pre/test', str(sub_id) + '/subject' + str(sub_id).zfill(2) + '.pkl')
with open(pickle_file_path, 'wb') as file:
    pickle.dump(dataset, file)

# 保存标签为CSV
csv_file_path = os.path.join('./data/pre/test', str(sub_id) + '/subject' + str(sub_id).zfill(2) + '.csv')
df = pd.DataFrame(dataset)
df.to_csv(csv_file_path, index=False)

print('Preprocessing and saving complete.')