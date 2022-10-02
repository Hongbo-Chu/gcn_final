from typing import Counter
import torch
import argparse
import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import os
import math
from torch.utils.data import DataLoader

from maintrain.models.gcn_test import GCN
from maintrain.models.gcn import graph_mlp as g_mlp
from maintrain.utils.datasets import wsi_dataset
from trainengine_modelparallel import train_one_wsi
from reduce_backbone import build_model
from maintrain.models.loss import myloss
from maintrain.utils.utils import merge_mini_patch, evaluate
from evaluate import evaluate_wsi

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_args_parser():
    parser = argparse.ArgumentParser(description='PyTorch implementation')
    parser.add_argument('--training_wsi', type=str, default="54",
                        help='which gpu to use if any (default: 0)')

    parser.add_argument('--wsi_folder', type=str, default="/root/autodl-tmp/wsis/wsi",
                        help='which gpu to use if any (default: 0)')

    parser.add_argument('--wsi_dict_path', type=str, default='/root/autodl-tmp/wsis/final_wsi.npy',
                        help='which gpu to use if any (default: 0)')

    parser.add_argument('--device0', type=str, default="cuda:0",
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--device1', type=str, default="cuda:1",
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--distributed', default=True, type=bool,
                        help='node rank for distributed training')
    parser.add_argument('--num_gpu', type=int, default=4,
                        help='which gpu to use if any (default: 0)')
    # 
    parser.add_argument('--backbone', type=str, default='vit',
                    help='backbonetype')
    parser.add_argument('--cluster_method', type=str, default='spectral',
                    help='cluster_method')
    parser.add_argument('--lr', type=float, default=1e-5,
                    help='backbonetype')
    parser.add_argument('--decay', type=float, default=0.1,
                    help='backbonetype')
    
    parser.add_argument('--cluster_num', type=int, default=9,
                help='backbonetype')
    
    parser.add_argument('--mask_decay_rate', type=float, default=0.1,
                help='backbonetype')
    
    parser.add_argument('--pos_choose', type=int, default=1,
                help='用于表示找物理特征的时候看周围几圈点')
    
    parser.add_argument('--mask_num', type=int, default=10,
                help='一个wsi加几轮mask')
    parser.add_argument('--mask_rate_high', type=float, default=0.1,
                help='高相似度点mask的比例')
    parser.add_argument('--mask_rate_mid', type=float, default=0.1,
                help='中相似度点mask的比例')
    parser.add_argument('--mask_rate_low', type=float, default=0.1,
                help='低相似度点mask的比例')
    parser.add_argument('--embeding_dim', type=int, default=768,
                help='嵌入维度')
    parser.add_argument('--epoch_per_wsi', type=int, default=20,
                help='一个wsi整体训练几轮')

    parser.add_argument('--mask_weight', type=float, default=0.1,
                help='loss中mask被增强的权重')
    parser.add_argument('--heirachi_clus_thres', type=int, default=10,
                help='层次聚类中距离的阈值')
    parser.add_argument('--edge_mask_inner', type=float, default=0.1,
                help='边mask中类内的比例')
    parser.add_argument('--edge_mask_inter', type=float, default=0.1,
                help='边mask类间的比例')
    parser.add_argument('--edge_mask_random', type=float, default=0.1,
                help='边mask随机的比例')
    parser.add_argument('--edge_enhance', type=float, default=1.0,
                help='对邻居边的增强')
    parser.add_argument('--weight_fold', type=float, default=0.2,
                help='在更新图的过程中当前轮数点的相似度，和上一轮中连接情况的比例')
    
    parser.add_argument('--log_folder', type=str, default='/root/autodl-tmp/debuging/runs/logs',
            help='日志存储文件夹')
    parser.add_argument('--weight_folder', type=str, default='/root/autodl-tmp/debuging/runs/weights',
            help='模型存储文件夹')
    parser.add_argument('--save_folder_test', type=str, default='/root/autodl-tmp/debuging/runs/test',
            help='模型存储文件夹')
    return parser


def save_acc(path, acc, epoch):
    save_path = os.path.join(path, 'eval' + '.md')
    if not os.path.exists(save_path):   
        with open(save_path, 'a+') as f:
            title =  '## acc log'+ '\n'
            f.write(title)
    with open(save_path, 'a+') as f:
        f.write(str(epoch) + ": acc=" + str(acc) + '\n')

# def crop_wsi(wsi_dict, wsi_tensor, patch_size, drop):
#     """
#     根据将一个大的wsi分层若干个size大小的小wsi
#     drop:当边角的块为中心块的%多少时，不会被丢掉
#     """
#     #先找出原wsi x, y 的最大值
#     x_max = 0
#     y_max = 0
#     print(type(wsi_dict))
#     for idx, patch in wsi_dict.items():
#         x, y = patch[2]
#         if x > x_max:
#             x_max = x
#         if y > y_max:
#             y_max = y
#     patch_edge_size = int(math.sqrt(patch_size))
#     print(patch_edge_size, x_max, y_max)
#     x_num = (x_max // patch_edge_size) + 1
#     y_num = (y_max // patch_edge_size) + 1
#     wsi_buffer = [([[]] * x_num) for i in range(y_num)]
#     tensor_buffer = [([[]] * x_num) for i in range(y_num)]
#     print(f"wsi,含有{len(wsi_dict)}块patches被拆分成[{x_num * y_num}]块")
#     for idx, patch in wsi_dict.items():
#         x_pos, y_pos = patch[2]
#         x = x_pos // patch_edge_size
#         y = y_pos // patch_edge_size

#         wsi_buffer[x][y].append(patch)
#         tensor_buffer[x][y].append(wsi_tensor[idx])
#     # 最后将二维的字典变成一维
#     wsi_list = []
#     tensor_list = []
#     for x_wsi, x_tensor in zip(wsi_buffer, tensor_buffer):
#         for wsi, ten in zip(x_wsi, x_tensor):
            
#             wsi_list.append(wsi)
#             tensor_list.append(torch.stack(ten))
#             wsi_list.append(wsi)
#     return wsi_list, tensor_list, x_num * y_num
    

    

    



def run():
    # Initialize distributed training context.
    
    # Define training and validation dataloader, copied from the previous tutorial
    # but with one line of difference: use_ddp to enable distributed data parallel
    # data loading.
    
#     model = Model(num_features, 128, num_classes).to(device)
    seed_everything(114514111)
    args = get_args_parser()
    args = args.parse_args()
    backboneModel = build_model(args.backbone).to(args.device0)
    graph_mlp = g_mlp(in_dim=5, hid_dim=16, out_dim = 1).to(args.device1)
    graph_model = GCN(in_dim=args.embeding_dim, num_hidden=128, out_dim=args.embeding_dim, num_layers=6, dropout=0,activation="prelu", residual=True,norm=nn.LayerNorm).to(args.device1)
    criterion = myloss().to(args.device1)
    
    #加载预训练参数
#     path = '../mae-multi3/output_pretrain_256/checkpoint-85.pth'
#     pretrain_model_dic = torch.load(path)
# #     print(list(backboneModel.state_dict().keys()))
#     #找共同的参数
#     same_param = {k:v for k, v in pretrain_model_dic['model'].items() if k in list(backboneModel.state_dict().keys())}
#     backboneModel.load_state_dict(same_param)
    # print(f"load params:{same_param.keys()}")
    for epoch in range(200):
        # img_load_path = os.path.join(args.wsi_folder, (args.training_wsi+ '.pt'))
        # dict_load_path = os.path.join(args.wsi_folder, (args.training_wsi + '.npy'))
        # wsi_img = torch.load(img_load_path)
        # wsi_dict =  dict(np.load(dict_load_path, allow_pickle='TRUE').item())
        # dict_crop, img_crop, total = crop_wsi(wsi_dict, wsi_img, args.batch_size, 1)
        # res_dict_list = []

        my_dataset = wsi_dataset(args.batch_size, args)
        wsi_loader = DataLoader(my_dataset, batch_size=1)

        for idx, (wdict, wimg) in enumerate(wsi_loader):
            #TODO 用collat_fn改一下size,现在wimg的大小是[1,n,c,h,w]
            print(f"统计一下真实标签数量")
            la = []
            wimg = wimg.squeeze(0)
            # for i in range(len(wdict)):
            #     la.append(wdict[i][3])
            # print(Counter(la))
            # clus_num = len(Counter(la))
            total =100
            res_dict = train_one_wsi(backboneModel, graph_model, graph_mlp, criterion, wimg, wdict, idx, total, epoch, args)
            res_dict_list.append(res_dict)
        #合并patch,并验证
        #每个patch返回{center_fae:[true_label]}
        a = merge_mini_patch(res_dict_list, 0.9)
        acc = evaluate(a)
        print(f"\n acc= {acc}\n")
        save_acc(args.save_folder_test, acc, epoch)
if __name__ == "__main__":
    run()