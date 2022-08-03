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

from maintrain.models.gcn_test import GCN
from maintrain.models.gcn import graph_mlp as g_mlp
from trainengine_modelparallel import train_one_wsi
from reduce_backbone import build_model
from maintrain.models.loss import myloss
from maintrain.utils.utils import merge_mini_patch, evaluate
from evaluate import evaluate_wsi


def get_args_parser():
    parser = argparse.ArgumentParser(description='PyTorch implementation')
    parser.add_argument('--training_wsi', type=str, default="43",
                        help='which gpu to use if any (default: 0)')

    parser.add_argument('--wsi_folder', type=str, default="/root/autodl-tmp/training_wsi",
                        help='which gpu to use if any (default: 0)')

    parser.add_argument('--device0', type=str, default="cuda:0",
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--device1', type=str, default="cuda:1",
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=1300,
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
    parser.add_argument('--cluster_method', type=str, default='hierarchy',
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
    parser.add_argument('--epoch_per_wsi', type=int, default=10,
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


    
    parser.add_argument('--log_folder', type=str, default='/root/autodl-tmp/7.26备份/runs/logs',
            help='日志存储文件夹')
    parser.add_argument('--weight_folder', type=str, default='/root/autodl-tmp/7.26备份/runs/weights',
            help='模型存储文件夹')
    parser.add_argument('--save_folder_test', type=str, default='/root/autodl-tmp/7.26备份/runs/test',
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

def crop_wsi(wsi_dict, wsi_tensor, size, drop):
    """
    根据将一个大的wsi分层若干个size大小的小wsi
    drop:当边角的块为中心块的%多少时，不会被丢掉
    """
    droplast = True
    wsi_num = len(wsi_dict)
    if wsi_num < size:
        return [wsi_dict], [wsi_tensor]
    else:
        crop_num = (wsi_num // size) + 1
        last = wsi_num % size
        if last/size > drop: #够大，不扔
            droplast = False
            print(f"lastsize:{last}")
        wsi_name = wsi_dict[0][0].split("_")[0]
        total = (crop_num - 1 if droplast else crop_num)
        print(f"wsi[{wsi_name}],(含有{wsi_num}块patches)被拆分成[{total}]块")
        wsi_buffer = []
        tensor_buffer =[]
        for i in range(crop_num):
            wsi_temp = {}
            tensor_temp = []
            if i == crop_num - 1:
                #是最后一组
                if droplast == False:
                    for idx, j in enumerate(range(i*size, wsi_num)):
                        wsi_temp[idx] = wsi_dict[j]    
                    tensor_temp = wsi_tensor[i*size:]
                    wsi_buffer.append(wsi_temp)
                    tensor_buffer.append(tensor_temp)
            else:
                for idx, j in enumerate(range(i*size, (1+i)*size)):
                    wsi_temp[idx] = wsi_dict[j]
                tensor_temp = wsi_tensor[i*size: (i+1)*size]
                wsi_buffer.append(wsi_temp)
                tensor_buffer.append(tensor_temp)
        return wsi_buffer, tensor_buffer, total
    



def run():
    # Initialize distributed training context.
    
    # Define training and validation dataloader, copied from the previous tutorial
    # but with one line of difference: use_ddp to enable distributed data parallel
    # data loading.
    
#     model = Model(num_features, 128, num_classes).to(device)
    args = get_args_parser()
    args = args.parse_args()
    backboneModel = build_model(args.backbone).to(args.device0)
    graph_mlp = g_mlp(in_dim=6, hid_dim=16, out_dim = 1).to(args.device1)
    graph_model = GCN(in_dim=args.embeding_dim, num_hidden=128, out_dim=args.embeding_dim, num_layers=6, dropout=0,activation="prelu", residual=True,norm=nn.LayerNorm).to(args.device1)
    criterion = myloss().to(args.device1)

    for epoch in range(200):
        img_load_path = os.path.join(args.wsi_folder, (args.training_wsi+ '.pt'))
        dict_load_path = os.path.join(args.wsi_folder, (args.training_wsi + '.npy'))
        wsi_img = torch.load(img_load_path)
        wsi_dict =  dict(np.load(dict_load_path, allow_pickle='TRUE').item())
        dict_crop, img_crop, total = crop_wsi(wsi_dict, wsi_img, args.batch_size, 0.5)
        res_dict_list = []
        for idx, (wdict, wimg) in enumerate(zip(dict_crop, img_crop)):
            print(f"统计一下真实标签数量")
            la = []
            for i in range(len(wdict)):
                la.append(wsi_dict[i][3])
            print(Counter(la))
            clus_num = len(Counter(la))
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