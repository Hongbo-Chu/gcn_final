import cv2
import torch
from torchvision import transforms
import torch
from tqdm import tqdm
import torch
import os
import numpy as np
import cv2
import torch
from torchvision import transforms

# def read_img(path):
#     transform = transforms.Compose([transforms.ToTensor()])
#     img = transform(cv2.imread(path))
#     return img


# # 先读取图片信息，再根据图片信息找tensor

# folder_path = r'/root/autodl-tmp/wsis/wsi'
# save_dict_path = '/root/autodl-tmp/save_wsi/wsi_dict.npy'
# save_dict = {}
# folder_clas = os.listdir(folder_path)
# print(folder_clas)
# for clas in folder_clas:
#     img_path = os.listdir(os.path.join(folder_path, clas))
#     for img in tqdm(img_path):
#         wsi_num = img.split("_")[0]
#         if str(wsi_num) not in save_dict.keys():
#             save_dict[str(wsi_num)] = []
#             save_dict[str(wsi_num)].append(img)
#             continue
#         save_dict[str(wsi_num)].append(img)

# #对于某一张单独的wsi:
# wsi_dict = {}
# for wsi_list in save_dict.items():
#     print(wsi_list[0])
#     x_min = 0
#     y_min = 0
#     x_true_list = []
#     y_true_list = []
#     for wsi_patch in wsi_list[1]:
#         x_true = int(wsi_patch.split("_")[3].split('.')[0])
#         y_true = int(wsi_patch.split("_")[4].split('.')[0])
#         x_true_list.append(x_true)
#         y_true_list.append(y_true)
#     x_min = min(x_true_list)
#     y_min = min(y_true_list)
#     x_pos = [(i - x_min) // 512 for i in x_true_list]
#     y_pos = [(i - y_min) // 512 for i in y_true_list]
#     wsi = []
#     for i in range(len(x_pos)):
#         wsi.append([wsi_list[1][i], (x_true_list[i], y_true_list[i]), x_pos[i], y_pos[i]])
#     wsi = sorted(wsi, key = lambda element: (element[2], element[3]))
#     wsi = [[item[0], item[1], (item[2], item[3])] for item in wsi]
#     wsi_dict[wsi_list[0]] = wsi
# print(wsi_dict.keys())
# np.save(save_dict_path, wsi_dict)

# # 根据npy生成对应的tensor
# aa = dict(np.load(save_dict_path, allow_pickle='TRUE').item())
# aa.keys()
# for k, wsi_patches in tqdm(aa.items()):
#     save_folder = r'/root/autodl-tmp/save_wsi'
#     wsi_tensor_list = []
#     wsi_npy_dict = {}
#     if k == '43' or k == '44':
#         continue
#     for idx, wsi_patch in tqdm(enumerate(wsi_patches)):
#         label = str(wsi_patch[0].split("_")[-3])
#         # print(label)
#         name = wsi_patch[0]
#         # print(name)
#         final_path = os.path.join(folder_path, label, name)
#         wsi_npy_dict[idx] = wsi_patch
#         try:
#             aa = read_img(final_path)
#         except:
#             print(final_path)
#         wsi_tensor_list.append(aa)
#     wsi_tensor = torch.stack(wsi_tensor_list)
#     torch.save(wsi_tensor,os.path.join(save_folder, str(k)+'.pth'))
#     np.save(os.path.join(save_folder, str(k)+'.npy'), wsi_npy_dict)


path = r'/root/autodl-tmp/save_wsi/43.pth'
aa = torch.load(path)
print(aa.size())