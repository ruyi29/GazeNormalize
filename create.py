import h5py
import numpy as np

imagData = np.zeros((30,3,128,256))
f = h5py.File("myh5py2.h5", "w")
f['data'] = imagData
f['labels'] = range(100)
# arr = np.arange(100)
# #只是单纯地想写一个进文件
# f.create_dataset("init", data=arr)
# #也可以引为一个对象    
# dset = f.create_dataset("init", data=arr)
# #同样也支持
# f["init"] = np.arange(100)

#在写完文件之后，一定记着要
f.close()