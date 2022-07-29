import torch
import argparse
import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optimizer
import tqdm
import os
from time import time
from reduce_backbone import build_model
from maintrain.utils.utils import Cluster
from maintrain.construct_graph import new_graph
from maintrain.utils.utils import chooseNodeMask, compute_pys_feature, fea2pos, chooseEdgeMask
from maintrain.models.gcn import GCN
from maintrain.utils.fold import fold_dict as fd
from maintrain.utils.fold import stable_dict as sd 
from maintrain.utils.fold import update_fold_dic
from maintrain.models.loss import myloss as ml
from maintrain.models.gcn import graph_mlp as g_mlp
# from train_engine_mp2 import train_one_wsi
from reduce_backbone import build_model
from evaluate import compute_acc, save_log_eval


def save_log(save_folder, wsi_name, mini_patch, total, epoch, mask_nodes, fold_dict, labels, time, center_fea, edge_fea, center_pos, edge_pos, loss, big_epoch, acc=None, train=True):
    """
    一个wsi存一个md
    """
    final_save_folder = os.path.join(save_folder, str(big_epoch))
    final_save_path = os.path.join(save_folder, str(big_epoch), (str(wsi_name) + '.md'))
    if not os.path.exists(final_save_folder):
        os.mkdir(final_save_folder)
    if not os.path.exists(final_save_path):   
        with open(final_save_path, 'a+') as f:
            title = '# wsi' + str(wsi_name) + ' training log'+ '\n'
            f.write(title)
            f.close()
    if train:
        with open(final_save_path, 'a+') as f:
            # title = '# wsi' + str(wsi_name) + ' mini_patch:' + str(mini_patch) + "/" + str() + 'traning time:' + str(time) + '\n'
            # f.write(title)
            center_map = set(center_fea).intersection(center_pos)
            edge_map = set(edge_fea).intersection(edge_pos)
            info = '## training    epoch:' + str(epoch) + '   mini_patch: ' + str(mini_patch) + "/" + str(total) + '   training_time: ' + str(time) + 's\n'
            f.write(info)
            f.write('### loss is: ' +str(float(loss)) + '\n')
            f.write("### labels: \n")
            f.write(str(labels))
            f.write("\n")
            f.write("### fold_nodes: \n")
            f.write(str(fold_dict.fold_dict))
            f.write('\n')
            f.write('### mask nodes: \n')
            f.write(str(mask_nodes))
            f.write('\n')
            a = '特征空间中心点共：' + str(len(center_fea)) + "个， 特征空间边缘点共：" + str(len(edge_fea)) + '个， 物理空间中心点共：' + str(len(center_pos)) + '个，物理空间边缘点共' + str(len(edge_pos)) + "个\n"
            f.write(a)
            f.write('中心对齐的点有: ' + str(len(center_map)) + "个， 边缘对齐的点有：" + str(len(edge_map)) + '个\n')
            f.write("***")
            f.write('\n')
            f.close()
    else:
        with open(final_save_path, 'a+') as f:
            center_map = set(center_fea).intersection(center_pos)
            edge_map = set(edge_fea).intersection(edge_pos)
            info = '## evaluate, epoch:' + str(epoch) + ' mini_patch: ' + str(mini_patch) + "/" + str() + ' training_time:' + str(time) + '\n'
            f.write(info)
            f.write('### acc is: ' +str(acc) + '\n')
            f.write("### labels: \n")
            f.write(str(labels))
            f.write("\n")
            f.write("### fold_nodes: \n")
            f.write(str(fold_dict.fold_dict))
            f.write('\n')
            f.write('### mask nodes: \n')
            f.write(str(mask_nodes))
            a = '特征空间中心点共：' + str(len(center_fea)) + "个， 特征空间边缘点共：" + str(len(edge_fea)) + '个， 物理空间中心点共：' + str(len(center_pos)) + '个，物理空间边缘点共' + str(len(edge_pos)) + "个\n"
            f.write(a)
            f.write('中心对齐的点有: ' + str(len(center_map)) + "个， 边缘对齐的点有：" + str(len(edge_map)) + '个')
            f.write("***")
            f.close()




def freeze(backbone, graph_model, args):
    """用于冻结模型的参数

    Args:
        backbone (torch.nn.Module): _description_
    """
    print("in this epoch the param of backbone is freezed")
    for child in backbone.children():
        for param in child.parameters():
            param.requires_grad = False
    #告诉优化器，哪些需要更新，那些不需要，这一步至关重要
    #filter() 函数用于过滤序列，过滤掉不符合条件的元素，返回由符合条件元素组成的新列表
    # optimizer.Adam(filter(lambda p: p.requires_grad, backbone.parameters()), lr=args.lr)
    return optimizer.Adam(list(graph_model.parameters()), lr=args.lr, weight_decay=args.decay)

def unfreeze(backbone, graph_model, args):
    """解冻模型的参数

    Args:
        backbone (torch.nn.Module): _description_
    """
    for child in backbone.children():
        for param in child.parameters():
            param.requires_grad = True
    return optimizer.Adam(list(backbone.parameters()) + list(graph_model.parameters()), lr=args.lr, weight_decay=args.decay)



def train_one_wsi(backbone: torch.nn.Module, gcn: torch.nn.Module, 
                    criterion:torch.nn.Module,
                    wsi_img,
                    wsi_dict,
                    mini_patch,
                    total,
                    big_epoch,
                    clus_num,
                    args=None):
            
   
    backbone.train()
    gcn.train()
    wsi_name = wsi_dict[0][0].split("_")[0]
    fold_dic = fd(args.batch_size)#用于记录被折叠的点
    stable_dic = sd()#用于记录稳定的点
    for epoch in range(args.epoch_per_wsi):
        start = time()
        print(f"wsi:[{wsi_name}], epoch:[{epoch}]:")
        input_img = wsi_img.to("cuda:0")

        if (epoch+1) % 3 == 0:
            optimizer = freeze(backbone, gcn, args)
        else:
            optimizer = unfreeze(backbone, gcn, args)

        #training
        node_fea = backbone(input_img)
        update_fold_dic(stable_dic, fold_dic)
         #先将折叠中心的node_fea添加
        for k in fold_dic.fold_dict.keys():# 不存在空的折叠点
            node_fea_k = torch.zeros(768).to("cuda:0")
            for node in fold_dic.fold_dict[k]:
                node_fea_k += node_fea[node]
            node_fea_k = node_fea_k / len(fold_dic.fold_dict[k])
            node_fea = torch.cat([node_fea, node_fea_k.unsqueeze(0)], dim = 0)

        node_fea_detach = node_fea.clone().detach()#从计算图中剥离
        # node_fea_detach = node_fea_detach.to("cpu")
        g, u_v_pair, edge_fea = new_graph(wsi_dict, fold_dic, node_fea_detach, 1, "cuda:1").init_graph()
        
        
        
        clu_label = Cluster(node_fea=node_fea_detach, cluster_num = clus_num, device='cuda:1').predict()
        for i in range(len(wsi_dict)):
            wsi_dict[i].append(clu_label[i])
        mask_rates = [args.mask_rate_high, args.mask_rate_mid, args.mask_rate_low]#各个被mask的比例
        mask_idx, fea_center, fea_edge, sort_idx_rst, cluster_center_fea = chooseNodeMask(node_fea_detach, clus_num, mask_rates, wsi_dict, "cuda:1", stable_dic, clu_label)#TODO 检查数量
        mask_edge_idx = chooseEdgeMask(u_v_pair, clu_label,sort_idx_rst, {"inter":0.1, "inner":0.1, "random":0.1} )#类内半径多一点
        node_fea[mask_idx] = 0
        edge_fea[mask_edge_idx] = 0
        print(f"this epoch mask nodes:{len(mask_idx)}, mask edges: {len(mask_edge_idx)}")
        g = g.to("cuda:1")
        edge_fea = edge_fea.to("cuda:1")
        node_fea = node_fea.to("cuda:1")
        # print(f"test发发发发发发{node_fea.size()}， {node_fea.device}, {edge_fea.size()}, {edge_fea.device}")
        predict_nodes = gcn(g, node_fea, edge_fea)

        loss = criterion(predict_nodes, clu_label, cluster_center_fea, mask_idx, 0.1, sort_idx_rst)
        # loss = criterion(predict_nodes, node_fea)
        print(f"epoch{[epoch]}, loss is:{[loss]}")
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        pys_center, pys_edge = compute_pys_feature(wsi_dict, args.pos_choose) #计算物理特征
        # fea2pos(fea_center, fea_edge, pys_center, pys_edge)#统计对齐信息并打印
        clu_labe_new = Cluster(node_fea=predict_nodes, cluster_num = clus_num, device='cuda:1').predict()
        for i in range(len(wsi_dict)):
            wsi_dict[i].pop(-1)
        train_time = int(time() - start)
        save_log(args.log_folder, wsi_name, mini_patch, total, epoch, mask_idx, fold_dic, clu_labe_new, train_time, fea_center, fea_edge, pys_center, pys_edge, loss, big_epoch, acc=None, train=True)
    true_label = []
    print("start eval")
    eval_start = time()
    for i in range(len(wsi_dict)):
        true_label.append(int(wsi_dict[i][3]))
    acc = compute_acc(true_label, clu_labe_new, fold_dic, args)
    print(f"acc={acc}")
    eval_time = time() - eval_start
    save_log_eval(args.save_folder_test, big_epoch, wsi_name, acc, mini_patch, eval_time, epoch, total)
        # save_log(args.log_folder, wsi_name, epoch, clu_label, mask_idx)
        #将这旧的聚类标签删除
