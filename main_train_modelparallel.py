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
from evaluate import evaluate_wsi


def get_args_parser():
    parser = argparse.ArgumentParser(description='PyTorch implementation')
    # 
    parser.add_argument('--device0', type=str, default="cuda:0",
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--device1', type=str, default="cuda:1",
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=300,
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
                help='一个wsi加几轮mask')
    parser.add_argument('--mask_rate_mid', type=float, default=0.1,
                help='一个wsi加几轮mask')
    parser.add_argument('--mask_rate_low', type=float, default=0.1,
                help='一个wsi加几轮mask')
    parser.add_argument('--embeding_dim', type=int, default=768,
                help='一个wsi加几轮mask')
    parser.add_argument('--epoch_per_wsi', type=int, default=10,
                help='一个wsi整体训练几轮')
    parser.add_argument('--log_folder', type=str, default='/root/autodl-tmp/7.26备份/runs/logs',
            help='日志存储文件夹')
    parser.add_argument('--weight_folder', type=str, default='/root/autodl-tmp/7.26备份/runs/weights',
            help='模型存储文件夹')
    parser.add_argument('--save_folder_test', type=str, default='/root/autodl-tmp/7.26备份/runs/test',
            help='模型存储文件夹')
    return parser


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
    # pretrained_dict  = torch.load('/root/autodl-tmp/mae-multi3/output_pretrain_256/checkpoint-30.pth')
    # backbone_dict = backboneModel.state_dict()
    # pretrained_dict  = {key: value for key, value in pretrained_dict.items() if (key in backbone_dict)}
    # print(pretrained_dict)
    graph_model = GCN(in_dim=args.embeding_dim, num_hidden=128, out_dim=args.embeding_dim, num_layers=6, dropout=0,activation="prelu", residual=True,norm=nn.LayerNorm).to(args.device1)
    # graph_mlp = g_mlp(in_dim=6, hid_dim=16, out_dim = 1).to('cuda:1')
    # optimizer = optim #.Adam(list(backboneModel.parameters()) + list(graph_model.parameters()) + list(graph_mlp.parameters()), lr=args.lr, weight_decay=args.decay)
    criterion = myloss().to(args.device1)
    # training_wsis = ['50', '44', '46', '48', '49', '57', '53', '56', '43']
    training_wsis = ['43']
    saving_path = '/root/autodl-tmp/training_wsi'
    # for epoch in range(200):
    #     for tw in training_wsis:
    #         img_load_path = os.path.join(saving_path, (tw + '.pt'))
    #         dict_load_path = os.path.join(saving_path, (tw + '.npy'))
    #         wsi_img = torch.load(img_load_path)
    #         wsi_dict =  dict(np.load(dict_load_path, allow_pickle='TRUE').item())
    #         dict_crop, img_crop, total = crop_wsi(wsi_dict, wsi_img, args.batch_size, 0.5)
    #         for idx, (wdict, wimg) in enumerate(zip(dict_crop, img_crop)):
    #             train_one_wsi(backboneModel, graph_model, graph_mlp, criterion, wimg, wdict, idx, total, epoch, args)
    #             evaluate_wsi(backboneModel, graph_model, graph_mlp, wimg, wdict,epoch, idx,  total, args)
    #             state = {'backbone': backboneModel.state_dict(), 'graph_mlp': graph_mlp.state_dict(), 'gcn': graph_model.state_dict()}
    #             save_path = os.path.join(args.weight_folder, 'epoch'+ str(epoch) + 'wsi' + str(tw) + '.pt')
    #             torch.save(state, save_path)
    for epoch in range(200):
        img_load_path = os.path.join(saving_path, ('43'+ '.pt'))
        dict_load_path = os.path.join(saving_path, ('43' + '.npy'))
        wsi_img = torch.load(img_load_path)
        wsi_dict =  dict(np.load(dict_load_path, allow_pickle='TRUE').item())
        dict_crop, img_crop, total = crop_wsi(wsi_dict, wsi_img, args.batch_size, 0.5)
        for idx, (wdict, wimg) in enumerate(zip(dict_crop, img_crop)):
            print(f"统计一下真实标签数量")
            la = []
            for i in range(len(wdict)):
                la.append(wsi_dict[i][3])
            print(Counter(la))
            clus_num = len(Counter(la))
            train_one_wsi(backboneModel, graph_model, criterion, wimg, wdict, idx, total, epoch, args)
        #合并patch,并验证
        #每个patch返回{center_fae:[true_label, num]}

            # evaluate_wsi(backboneModel, graph_model, wimg, wdict,epoch, idx,  total, clus_num, args)
            # state = {'backbone': backboneModel.state_dict(), 'gcn': graph_model.state_dict()}
            # save_path = os.path.join(args.weight_folder, 'epoch'+ str(epoch) + 'wsi' + str(43) + 'minipatch' + str(idx) + '.pt')
            # torch.save(state, save_path)
if __name__ == "__main__":
    run()