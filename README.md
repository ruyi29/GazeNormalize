# GazeNormalize
主体代码在pretreat.py和createh5.py文件中

## pretreat.py
对数据进行归一化，并将结果数据存储在pre文件夹中

## createh5.py
将归一化后的数据转化为h5文件格式

h5文件中包含的数据有：
- frame_index: 
- face_patch: 尺寸为224*224的人脸图像
- face_mat_norm: 数据归一化期间的旋转矩阵
- face_gaze: 归一化的凝视方向，尺寸为二维，包括水平和垂直凝视方向

## dataset
data中的raw文件夹中存储未经预处理的原始数据，pre文件夹中存储经过归一化后的数据，h5文件夹中存储转化成h5文件后的数据

文件格式如下：

    data
    ├─raw
    |  ├─train
    |  |   ├─ 4
    |  |   |  ├─ Photo
    |  |   |  |    └─ 1~2376.jpg
    |  |   |  ├─ Video
    |  |   |  |    └─ 1~9.avi
    |  |   |  ├─ coordinate.txt
    |  |   |  └─ light.txt
    |  |   ├─ 5
    |  |   |  └─ ...
    |  |   └─ ...
    │  └─test
    |      └─...
    |
    ├─pre
    |  ├─train
    |  |   ├─ 4
    |  |   |  ├─preprocessed_images
    |  |   |  |       ├─ preprocessed_image_1.jpg
    |  |   |  |       └─ ...
    |  |   |  ├─subject04.csv
    |  |   |  └─subject04.pkl
    |  |   ├─ 5
    |  |   |  └─ ...
    |  |   └─ ...
    │  └─test
    |
    ├─h5
    |  ├─train
    |  |   ├─ subject04.h5
    |  |   └─ ...
    │  └─test
    |
    └─train_test_split.json