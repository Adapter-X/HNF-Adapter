import cv2
import numpy as np
import os

# 文件夹路径
# folder_path = '/media/dl_shouan/ZHITAI/Adapter/data/segmentation/Oxford-IIIT_adapter/annotations/GT_seg'
folder_path = '/media/dl_shouan/ZHITAI/Adapter/data/segmentation/Oxford-IIIT_adapter/images'

# 新建的文件夹路径
new_folder_path = '/media/dl_shouan/ZHITAI/Adapter/data/segmentation/Oxford-IIIT_adapter/images_2'

# 如果新的文件夹不存在，创建它
if not os.path.exists(new_folder_path):
    os.makedirs(new_folder_path)

# 获取文件夹中的所有文件
files = os.listdir(folder_path)

# 对于文件夹中的每个文件
for file in files:
    if file.endswith('.png') or file.endswith('.jpg'):
        img = cv2.imread(os.path.join(folder_path, file))
        #resize image to 224x224
        img = cv2.resize(img, (512, 512))
        # img = img[:, :, 0]

        cv2.imwrite(os.path.join(new_folder_path, file), img)