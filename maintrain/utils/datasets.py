import torch
from torch.utils.data import Dataset
import cv2
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import math
import os
"""
所有的wsi的信息都存储在一个npy文件里面，
文件形式{wsi_idx:{idx:[]}}

"""

def wsi_transform():
    return transforms.Compose([transforms.ToTensor()])
class wsi_dataset(Dataset):
    '''从一个wsi切边中提取'''
    def __init__(self, patch_size, args=None) -> None:
        super(Dataset, self).__init__()
        #指定某张wsi
        self.wsi_idx = args.training_wsi
        #该张wsi对应的patch信息
        self.wsi_dict = dict(np.load(args.wsi_dict_path, allow_pickle='TRUE').item())[self.wsi_idx]
        #将对应的patch信息裁剪生成一个个大patch列表
        self.crop_wsi(patch_size)
        self.folder_path = args.wsi_folder
        self.transforms = wsi_transform()
        
    def __getitem__(self, index):
        patch_tobe_process = self.patch_list[index]
        img_ten = self.get_img(patch_tobe_process)
        return patch_tobe_process, img_ten
    def __len__(self):
        return len(self.patch_list)
    def crop_wsi(self, patch_size):
        x_max = 0
        y_max = 0
        for idx, patch in self.wsi_dict.items():
            x, y = patch[2]
            # print(patch)
            # assert False
            if x > x_max:
                x_max = x
            if y > y_max:
                y_max = y
        patch_edge_size = int(math.sqrt(patch_size))
        print(patch_edge_size, x_max, y_max)
        x_num = (x_max // patch_edge_size) + 1
        y_num = (y_max // patch_edge_size) + 1
        wsi_buffer = [[[] for j in range(x_num)] for i in range(y_num)]
        print(f"wsi,含有{len(self.wsi_dict)}块patches被拆分成[{x_num * y_num}]块")
        for idx, patch in self.wsi_dict.items():
            x_pos, y_pos = patch[2]
            x = x_pos // patch_edge_size
            y = y_pos // patch_edge_size
            try:
                wsi_buffer[y][x].append(patch)
            except:
                print(x, y)
                print(x_num, y_num)
                assert False, 'out of range'
        # 最后将二维的字典变成一维
        self.patch_list = []
        for x_wsi in wsi_buffer:
            for wsi in x_wsi:           
                if wsi != []:
                    self.patch_list.append(wsi)
                    print(len(wsi))

    def get_img(self, patch):
        """
        读取一个大patch的图片，并返回对应的tensor[N,C,H,W]
        args:
            patch(type:list):一个大patch的list信息
        retunrs:
            img_tensor: [N,C,H,W]
        """
        def get_path_from_name(name):
            """根据图片的名字生成图片的地址"""
            label = str(name.split("_")[-3])
            # print(label)
            # print(name)
            final_path = os.path.join(self.folder_path, label, name)
            return final_path
        
        def read_img(path, transforms):
            img = transforms(cv2.imread(path))
                # assert False, "csao"
            return img
        img_tensor_list = [] #用于存放所有的img_tensor
        for img in patch:
            name = img[0]
            img_path = get_path_from_name(name)
            img_tensor = read_img(img_path, self.transforms)
            img_tensor_list.append(img_tensor)
        return torch.stack(img_tensor_list)
        

if __name__ == '__main__':
    my_dataset = wsi_dataset(6000)
    my_loader = DataLoader(my_dataset,batch_size=1)
    for i, j in my_loader:
        print(type(i))
        print(type(j))
        print(j.size())
        assert False



        