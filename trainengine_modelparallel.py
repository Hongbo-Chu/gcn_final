import torch
import torch.nn.functional as F
import torch.optim as optimizer
import os
from time import time
from collections import Counter
from maintrain.utils.utils import Cluster
from maintrain.construct_graph import new_graph
from maintrain.utils.utils import chooseNodeMask, compute_pys_feature, fea2pos, chooseEdgeMask, compute_clus_center
from maintrain.utils.fold import stable_dict as sd
# from maintrain.utils.fold import update_fold_dic


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

def unfreeze(backbone, graph_model, graph_mlp, args):
    """解冻模型的参数

    Args:
        backbone (torch.nn.Module): _description_
    """
    for child in backbone.children():
        for param in child.parameters():
            param.requires_grad = True
    return optimizer.Adam(list(backbone.parameters()) + list(graph_model.parameters()) + list(graph_mlp.parameters()), lr=args.lr, weight_decay=args.decay)



def train_one_wsi(backbone: torch.nn.Module, gcn: torch.nn.Module, 
                    graph_mlp,
                    criterion:torch.nn.Module,
                    wsi_img,
                    wsi_dict,
                    mini_patch,
                    total,
                    big_epoch,
                    args=None):
            
    backbone.train()
    gcn.train()
    wsi_name = wsi_dict[0][0].split("_")[0]
    stable_dic = sd(3)#用于记录被折叠的点
    for epoch in range(args.epoch_per_wsi):

        debug_path = '/root/autodl-tmp/debuging/fold2.txt'
        with open(debug_path, 'a+') as f:
            f.write('minipatch:' + str(mini_patch) + 'epoch:' + str(epoch))
            f.write('\n')
        start = time()
        print(f"wsi:[{wsi_name}], epoch:[{epoch}]:")
        input_img = wsi_img.to(args.device0)

        if (epoch+1) % 3 == 0:
            optimizer = freeze(backbone, gcn, graph_mlp, args)
        else:
            optimizer = unfreeze(backbone, gcn, graph_mlp, args)


        #training
        # print(input_img.size())
        # assert False, "fff"
        # kk = torch.randn([64, 3, 224, 224]).to(args.device0)

        node_fea = backbone(input_img)
        node_fea_detach = node_fea.clone().detach()#从计算图中剥离

        # node_fea_detach = node_fea_detach.to("cpu")
        g, u_v_pair, edge_fea = new_graph(wsi_dict, stable_dic, node_fea_detach, args.edge_enhance, graph_mlp, args.device1).init_graph(args)
        
        
        print(node_fea_detach.size())
        # assert False, '111ws'
        clu_label, clus_num = Cluster(node_fea=node_fea_detach.cpu(), device=args.device1, method=args.cluster_method).predict1(num_clus=2)
        stable_dic.update_fold_dic(node_fea_detach, clus_num)
         #先将折叠中心的node_fea变更
        for k in stable_dic.fold_dict.keys():
            node_fea[k] = stable_dic.fold_node_fea[k]
        
        for i in range(len(wsi_dict)):
            wsi_dict[i].append(clu_label[i])
        mask_rates = [args.mask_rate_high, args.mask_rate_mid, args.mask_rate_low]#各个被mask的比例
        # print(f"检查检查{fold_dic.stable_dic.keys()}")
        mask_idx, fea_center, fea_edge, sort_idx_rst, cluster_center_fea = chooseNodeMask(node_fea_detach, clus_num, mask_rates, wsi_dict, args.device1, stable_dic, clu_label)#TODO 检查数量
        # print(f"更新之后？？{stable_dic.stable_dic}")
        mask_edge_idx = chooseEdgeMask(u_v_pair, clu_label,sort_idx_rst, {"inter":args.edge_mask_inter, "inner":args.edge_mask_inner, "random": args.edge_mask_random} )#类内半径多一点
        node_fea[mask_idx] = 0
        edge_fea[mask_edge_idx] = 0
        print(f"this epoch mask nodes:{len(mask_idx)}, mask edges: {len(mask_edge_idx)}")
        g = g.to(args.device1)
        edge_fea = edge_fea.to(args.device1)
        node_fea = node_fea.to(args.device1)
        predict_nodes = gcn(g, node_fea, edge_fea)
        # print("*"*100)
        # print('\n')
        # print(predict_nodes.size())
        loss = criterion(predict_nodes, clu_label, cluster_center_fea, mask_idx, args.mask_weight, sort_idx_rst)
        # loss = criterion(predict_nodes, node_fea)
        print(f"epoch{[epoch]}, loss is:{[loss]}")
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        pys_center, pys_edge = compute_pys_feature(wsi_dict, args.pos_choose) #计算物理特征
        # fea2pos(fea_center, fea_edge, pys_center, pys_edge)#统计对齐信息并打印
        predict_nodes_detach = predict_nodes.clone().detach()
        clu_labe_new = Cluster(node_fea=predict_nodes_detach.cpu(), device=args.device1, method=args.cluster_method).predict2(num_clus=clus_num)
        for i in range(len(wsi_dict)):
            wsi_dict[i].pop(-1)
        train_time = int(time() - start)
        clus_centers = compute_clus_center(predict_nodes_detach, clu_labe_new)
        
        #统计一下真实标签的数量
        la = []
        for i in range(len(wsi_dict)):
            la.append(wsi_dict[i][3])
        true_label = dict(Counter(la))
        save_log(args.log_folder, wsi_name, mini_patch, total, epoch, mask_idx, stable_dic, clu_labe_new, train_time, fea_center, fea_edge, pys_center, pys_edge, loss, big_epoch, acc=None, train=True, true_clus_num=true_label, clus_num=clus_num)
    
    true_label = []
    for i in range(len(wsi_dict)):
        true_label.append(int(wsi_dict[i][3]))
    clus_centers = compute_clus_center(predict_nodes_detach, clu_labe_new)
    res_dict = {center:[] for center in clus_centers}
    for idx, t_label in enumerate(true_label):
        res_dict[clus_centers[clu_labe_new[idx]]].append(t_label)
    return res_dict
    