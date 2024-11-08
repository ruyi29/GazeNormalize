# 遍历文件夹里的文件名

import os

def get_folder_names(directory):
    # 获取指定目录下的所有文件和文件夹名称
    all_items = os.listdir(directory)
    folder_names = [int(item) for item in all_items if os.path.isdir(os.path.join(directory, item))]
    folder_names.sort()
    return folder_names

# 示例用法
directory_path = './data/raw/test/'  # 替换为你的文件夹路径
folders = get_folder_names(directory_path)
print(folders)