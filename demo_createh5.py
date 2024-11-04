import os
import cv2
import numpy as np
import h5py
import warp_norm
import pandas as pd
import ast

sub_id = 36
image_folder = './data/pre/test/'+str(sub_id)+'/preprocessed_images'
csv_file_path = './data/pre/test/'+str(sub_id)+'/'+'subject' + str(sub_id).zfill(2) + '.csv'
df = pd.read_csv(csv_file_path, index_col='image_path')
image_num = df.shape[0]
print("image_num: ", image_num)

# 保存为HDF5文件
hdf_fpath = os.path.join('./data/h5/test', 'subject' + str(sub_id).zfill(2) + '.h5')
output_h5_id = h5py.File(hdf_fpath, 'w')
print('output file save to ', hdf_fpath)

save_index = 0
output_frame_index = []
output_face_patch = []
output_face_mat_norm = []
output_face_gaze = []

output_frame_index = output_h5_id.create_dataset("frame_index", shape=(image_num, 1),
                                                            dtype=int, chunks=(1, 1))
output_face_patch = output_h5_id.create_dataset("face_patch", shape=(image_num, 224, 224, 3),
                                                compression='lzf', dtype=np.uint8,
                                                chunks=(1, 224, 224, 3))
output_face_mat_norm = output_h5_id.create_dataset("face_mat_norm", shape=(image_num, 3, 3),
                                                dtype=float, chunks=(1, 3, 3))
output_face_gaze = output_h5_id.create_dataset("face_gaze", shape=(image_num, 2),
                                                dtype=float, chunks=(1, 2))

# 遍历文件假
for image_path, _, mat_norm, gc_normalized in df.itertuples(index=True):
    image = cv2.imread(image_folder + "/" + image_path)
    frame_index = int(''.join(filter(str.isdigit, image_path)))
    mat_norm_array = np.fromstring(mat_norm.replace('\n', ' ').replace('[', '').replace(']', ''), sep=' ').reshape(3, 3)

    gc_normalized_array = np.fromstring(gc_normalized.replace('\n', ' ').replace('[', '').replace(']', ''), sep=' ').reshape(1, 3)
    # print("gc_normalized_array: ", gc_normalized_array)
    # exit()
    gaze_theta = np.arcsin(gc_normalized_array[0][1])
    gaze_phi = np.arctan2(gc_normalized_array[0][0], gc_normalized_array[0][2])
    gaze_norm_2d = np.asarray([gaze_theta, gaze_phi])
    face_gaze = gaze_norm_2d.reshape(2)

    # print("frame_index: ", frame_index)

    output_frame_index[save_index] = frame_index
    output_face_patch[save_index] = image
    output_face_mat_norm[save_index] = mat_norm_array
    output_face_gaze[save_index] = face_gaze

    save_index += 1

output_h5_id.close()
print('close the h5 file')
print('finish the subject: ', sub_id)    
  

