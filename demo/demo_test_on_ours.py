import os
import cv2
import csv
import dlib
from imutils import face_utils
import numpy as np
import torch
from torchvision import transforms
from model import gaze_network
import warp_norm_simple
import pandas as pd
import utils
import time
import argparse

def read_csv_parameters(csvpath,w,h):
    with open(csvpath, 'r', encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        gcs=[]
        ridlist=[]
        formal=0.0
        for i, row in enumerate(reader):
            x = row['gazeX']
            y = row['gazeY']
            if x==formal:
                ridlist.append(len(gcs))
            if len(gcs) >= 0:
                formal = row['gazeX']
            gcs.append(np.array([float(x)*int(w), float(y)*int(h)]))
        return gcs,ridlist



trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),  # this also convert pixel value from [0,255] to [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
cam_ousrs= './testpart/camOurs.xml'
fs_ours = cv2.FileStorage(cam_ousrs, cv2.FILE_STORAGE_READ)
w_ours = 960
h_ours = 540
camera_matrix_ours = fs_ours.getNode('Camera_Matrix').mat() # camera calibration information is used for data normalization
camera_distortion_ours = fs_ours.getNode('Distortion_Coefficients').mat()



if __name__ == '__main__':

    numlist = [3, 5, 24]
    for num in numlist:

        video_path = f"D:/drive_vision/test01/webcam_face_{num}.avi"
        csv_path= f"D:/drive_vision/test01/gaze_{num}.csv"
        save_path = f'./test/estimation/Gaze360_{num}.csv'
        pre_trained_model_path = './ckpt/epoch_15_0.0001_ckpt.pth.tar'
        # pre_trained_model_path = r"D:\drive_vision\res\Xgaze\448_single_stride_2\epoch_15_0.0001_ckpt.pth.tar"
        # #epoch_15_0.0001_ckpt.pth.tar  #epoch_24_ckpt.pth.tar
        camera_matrix = camera_matrix_ours
        camera_distortion = camera_distortion_ours
        w = w_ours
        h = h_ours

        gcs,ridlist=read_csv_parameters(csv_path,w,h)
        ckpt = torch.load(pre_trained_model_path, map_location=torch.device('cpu'))
        model = gaze_network()
        model.load_state_dict(ckpt['model_state'], strict=True)  # load the pre-trained model
        face_model_load = np.loadtxt('./modules/face_model.txt')
        # model.eval()  # change it to the evaluation mode
        predictor, face_detector = warp_norm_simple.xmodel()

        cap=cv2.VideoCapture(video_path)
        _, image = cap.read() #只取视频第一帧
        elist=[]
        count=0
        i=1
        flag=ridlist[0]
        with open(save_path, 'w',newline='') as csvf:
            writer = csv.writer(csvf)
            writer.writerow(['num', 'vecter'])
            while count<200:
                print(count)
                gc=gcs[count]
                img_normalized, gcn,face_count= warp_norm_simple.GazeNormalization(image, camera_matrix,
                                                                        camera_distortion,
                                                                       gc, w, h, predictor,
                                                                       face_detector,face_model_load)


                if face_count==0:
                    count += 1
                    _, image = cap.read()
                    continue
                if count==flag:
                    flag=ridlist[i]
                    i=i+1
                    count += 1
                    _, image = cap.read()
                    continue
                input_var = img_normalized[:, :, [2, 1, 0]]  # from BGR to RGB
                input_var = trans(input_var)
                input_var = torch.autograd.Variable(input_var.float())
                input_var = input_var.view(1, input_var.size(0), input_var.size(1),
                                           input_var.size(2))  # the input must be 4-dimension
                pred_gaze = model(input_var)  # get the output gaze direction, this is 2D output as pitch and raw rotation
                pred_gaze = pred_gaze[0]  # here we assume there is only one face inside the image, then the first one is the prediction
                pred_gaze_np = pred_gaze.cpu().data.numpy()  # convert the pytorch tensor to numpy array
                e = utils.angular_error(np.array([pred_gaze_np]), warp_norm_simple.vector_to_pitchyaw(np.array([gcn])))[0]
                print('error:', e)
                elist.append(e)
                _, image = cap.read()
                count+=1
                writer.writerow([count,elist[-1]])

        csvf.close()
        print(elist)
        print(np.mean(elist))
