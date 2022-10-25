from cProfile import label
import torch
import torch.nn.functional as F
import torch.optim as optimizer
import os
import numpy as np
from time import time
from collections import Counter
from maintrain.utils.utils import Cluster
from maintrain.construct_graph import new_graph
from maintrain.utils.utils import chooseNodeMask, compute_pys_feature, fea2pos, chooseEdgeMask, compute_clus_center
from maintrain.utils.fold import stable_dict as sd
from maintrain.utils.train_eval import evaluate
from collections import Counter
# from maintrain.utils.fold import update_fold_dic

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



from dgl.nn import GraphConv

class myGCN(torch.nn.Module):
    def __init__(self, in_feats, num_classes):
        super(myGCN, self).__init__()
        self.conv1 = GraphConv(in_feats, 512)
        self.conv2 = GraphConv(512, 256)
        self.conv3 = GraphConv(256, 128)
        self.conv4 = GraphConv(128, 64)
        self.conv5 = GraphConv(32, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h





def train_eval(pre, label):

    # 用一个cluster中最多的种类作为这个cluster的label
    buffer = [[] for _ in range(len(set(pre)))]
    for i in range(len(pre)):
        buffer[pre[i]].append(int(label[i]))
    label = np.array(label)
    true_num = 0
    for clu in buffer:
        aa = dict(Counter(clu))
        print(aa)
        temp = sorted(dict(Counter(clu)).items(), key=lambda x: x[1], reverse=True)
        true_num += temp[0][1]
    acc = true_num / len(pre)
    # _, _, _, acc = evaluate(label, pre)
    print(f"acc:[{acc}]")
    return acc


def save_log(save_folder, wsi_name, mini_patch, total, epoch, mask_nodes, fold_dict, labels, time, center_fea, edge_fea, center_pos, edge_pos, loss, big_epoch, acc=None, train=True, true_clus_num = None, clus_num = None):
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
            clus_info = '### 真实情况：' + str(true_clus_num)+ ', 实际聚类的时候分成了' + str(clus_num) + '类 \n'
            f.write(clus_info)
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




def freeze(backbone, graph_model, graph_mlp, args):
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
    return optimizer.Adam(list(graph_model.parameters()) + list(graph_mlp.parameters()), lr=args.lr, weight_decay=args.decay)

def unfreeze(graph_model, graph_mlp, args):
    """解冻模型的参数

    Args:
        backbone (torch.nn.Module): _description_
    """
    return optimizer.Adam(list(graph_model.parameters()) + list(graph_mlp.parameters()), lr=args.lr, weight_decay=args.decay)



def train_one_wsi1(backbone: torch.nn.Module, gcn: torch.nn.Module, 
                    graph_mlp,
                    criterion:torch.nn.Module,
                    wsi_img,
                    wsi_dict,
                    mini_patch,
                    total,
                    big_epoch,
                    args=None):
            
    backbone.eval()
    gcn.train()
    print(wsi_dict[0])
    wsi_name = str(wsi_dict[0][0]).split("_")[0]
    stable_dic = sd(3)#用于记录被折叠的点
    optimizer = unfreeze(gcn, graph_mlp, args)
    for epoch in range(args.epoch_per_wsi):

        debug_path = '/mnt/cpath2/lf/data/fold2.txt'
        with open(debug_path, 'a+') as f:
            f.write('minipatch:' + str(mini_patch) + 'epoch:' + str(epoch))
            f.write('\n')
        start = time()
        print(f"wsi:[{wsi_name}], epoch:[{epoch}]:")
        input_img = wsi_img.to(args.device0)

        # if (epoch+1) % 3 == 0:
        #     optimizer = freeze(backbone, gcn, graph_mlp, args)
        # else:
        

        # node_fea = backbone(input_img)

        #training
        # print(input_img.size())
        # assert False, "fff"
        # kk = torch.randn([64, 3, 224, 224]).to(args.device0)
        #分着来 
        buffer = []
        bs = input_img.size()[0]
        single_step = 32
        start = 0
        with torch.no_grad():
            for i in range(int(bs / single_step) + 1):
                if start + single_step < bs:
                    _, node_fea, _ = backbone(input_img[start:start+single_step], input_img[start:start+single_step],input_img[start:start+single_step], 0.75)
                else:
                    _, node_fea, _ = backbone(input_img[start:], input_img[start:],input_img[start:], 0.75)
                start += single_step
                buffer.extend(node_fea)
            node_fea = torch.stack(buffer)
    
        node_fea_detach = node_fea.clone().detach()#从计算图中剥离
        # node_fea_detach = node_fea_detach.to("cpu")
        g, u_v_pair, edge_fea = new_graph(wsi_dict, stable_dic, node_fea_detach, args.edge_enhance, graph_mlp, args.device1).init_graph(args)
        
        
        # print(node_fea_detach.size())
        # # assert False, '111ws'
        # clu_label, clus_num = Cluster(node_fea=node_fea_detach.cpu(), device=args.device1, method=args.cluster_method).predict1(num_clus=2)
        # stable_dic.update_fold_dic(node_fea_detach, clus_num)
        #  #先将折叠中心的node_fea变更
        # for k in stable_dic.fold_dict.keys():
        #     node_fea[k] = stable_dic.fold_node_fea[k]
        labels = []
        for i in range(len(wsi_dict)):
            # wsi_dict[i].append(clu_label[i])
            labels.append(int(wsi_dict[i][3]))
        # mask_rates = [args.mask_rate_high, args.mask_rate_mid, args.mask_rate_low]#各个被mask的比例
        # # print(f"检查检查{fold_dic.stable_dic.keys()}")
        # mask_idx, fea_center, fea_edge, sort_idx_rst, cluster_center_fea = chooseNodeMask(node_fea_detach, clus_num, mask_rates, wsi_dict, args.device1, stable_dic, clu_label)#TODO 检查数量
        # # print(f"更新之后？？{stable_dic.stable_dic}")
        # mask_edge_idx = chooseEdgeMask(u_v_pair, clu_label,sort_idx_rst, {"inter":args.edge_mask_inter, "inner":args.edge_mask_inner, "random": args.edge_mask_random} )#类内半径多一点
        # node_fea[mask_idx] = 0
        # edge_fea[mask_edge_idx] = 0
        # print(f"this epoch mask nodes:{len(mask_idx)}, mask edges: {len(mask_edge_idx)}")

        g = g.to(args.device1)
        # edge_fea = edge_fea.to(args.device1)
        node_fea = node_fea.to(args.device1)
        logits = gcn(g, node_fea)
        pred = logits.argmax(1)
        labels = torch.tensor(labels).to(args.device1)
        total_len = len(labels)
        train_num = int(0.8 * total_len)
        loss = F.cross_entropy(logits[:train_num], labels[:train_num])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_acc = (pred[:train_num] == labels[:train_num]).float().mean()
        eval_acc = (pred[train_num + 1:] == labels[train_num + 1:]).float().mean()
        print(f"train_acc={train_acc}, train_loss={loss}, eval_acc = {eval_acc}")








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
from reduce_backbone import build_model
from maintrain.models.loss import myloss
from maintrain.utils.utils import merge_mini_patch, evaluate
from evaluate import evaluate_wsi
import models_mae




def get_args_parser():
    parser = argparse.ArgumentParser(description='PyTorch implementation')
    parser.add_argument('--training_wsi', type=str, default="43",
                        help='which gpu to use if any (default: 0)')

    parser.add_argument('--wsi_folder', type=str, default="/mnt/cpath2/lf/data/wsis",
                        help='which gpu to use if any (default: 0)')

    parser.add_argument('--wsi_dict_path', type=str, default='/mnt/cpath2/lf/data/final_wsi_2.npy',
                        help='which gpu to use if any (default: 0)')

    parser.add_argument('--device0', type=str, default="cuda:3",
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--device1', type=str, default="cuda:5",
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=6000,
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
    parser.add_argument('--weight_fold', type=float, default=0.2,
                help='在更新图的过程中当前轮数点的相似度，和上一轮中连接情况的比例')
    
    parser.add_argument('--log_folder', type=str, default='/mnt/cpath2/lf/gcn_wsi/gcn_final-main/runs/logs',
            help='日志存储文件夹')
    parser.add_argument('--weight_folder', type=str, default='/mnt/cpath2/lf/gcn_wsi/gcn_final-main/runs/weights',
            help='模型存储文件夹')
    parser.add_argument('--save_folder_test', type=str, default='/mnt/cpath2/lf/gcn_wsi/gcn_final-main/runs/test',
            help='模型存储文件夹')
    parser.add_argument('--save_folder_train', type=str, default='/mnt/cpath2/lf/gcn_wsi/gcn_final-main/runs/save',
            help='模型存储文件夹')



    # parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL')
    parser.add_argument('--instance_temperature', default=0.5, type=int)
    parser.add_argument('--cluster_temperature', default=1, type=int)
    parser.add_argument('--num_class', default=9, type=int)
    parser.add_argument('--representation_size', default=768, type=int)
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)
    return parser




def main():
    seed_everything(114514111)
    args = get_args_parser()
    args = args.parse_args()
    # backboneModel = build_model(args.backbone).to(args.device0)
    backboneModel = models_mae.__dict__['mae_vit_base_patch16'](norm_pix_loss=args.norm_pix_loss,
                                            args=args)
    backboneModel.to(args.device0)
    graph_mlp = g_mlp(in_dim=5, hid_dim=16, out_dim = 1).to(args.device1)
    graph_model = myGCN(768, 9).to(args.device1)
    criterion = myloss(args).to(args.device1)
    path = '/mnt/cpath2/lf/mae_multi/output_pretrain/checkpoint-138.pth'
    checkpoint = torch.load(path)
    print('start load cheakpoint...')
    backboneModel.load_state_dict(checkpoint['model'])
    # graph_model.load_state_dict(checkpoint['gcn'])
    # graph_mlp.load_state_dict(checkpoint['graph_mlp'])
    print(args.batch_size)
    print("load over")
    #加载预训练参数
    for epoch in range(200):
        res_dict_list = []

        my_dataset = wsi_dataset(args.batch_size, args)
        wsi_loader = DataLoader(my_dataset, batch_size=1)

        for idx, (wdict, wimg) in enumerate(wsi_loader):
            #TODO 用collat_fn改一下size,现在wimg的大小是[1,n,c,h,w]
            print(f"统计一下真实标签数量")
            la = []
            wimg = wimg.squeeze(0)
            for i in range(len(wdict)):
                la.append(int(wdict[i][3]))
            print(Counter(la))
            clus_num = len(Counter(la))
            total =100
            res_dict = train_one_wsi1(backboneModel, graph_model, graph_mlp, criterion, wimg, wdict, idx, total, epoch, args)
            res_dict_list.append(res_dict)
        #合并patch,并验证
        #每个patch返回{center_fae:[true_label]}
        a = merge_mini_patch(res_dict_list, 0.9)
        acc = evaluate(a)
        print(f"\n acc= {acc}\n")
        save_acc(args.save_folder_test, acc, epoch)





if __name__ == "__main__":
    main()