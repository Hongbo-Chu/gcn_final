# from matplotlib.font_manager import _Weight
from cProfile import label
from collections import Counter
import torch
import numpy as np
from sklearn import metrics
from munkres import Munkres
from sklearn.metrics import f1_score

from time import time
import os
from reduce_backbone import build_model
from maintrain.utils.utils import Cluster
from maintrain.construct_graph import new_graph
from maintrain.utils.utils import chooseNodeMask, compute_pys_feature, fea2pos, chooseEdgeMask
from maintrain.models.gcn import GCN
from maintrain.utils.fold import fold_dict as fd
from maintrain.utils.fold import stable_dict as sd
from maintrain.models.loss import myloss as ml
from maintrain.models.gcn import graph_mlp as g_mlp
# from train_engine_mp2 import train_one_wsi
from reduce_backbone import build_model
from maintrain.utils.fold import update_fold_dic#test
def save_log_eval(save_folder, big_epoch, wsi_name, acc, mini_patch, time, epoch, total):
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
   
    with open(final_save_path, 'a+') as f:
        # title = '# wsi' + str(wsi_name) + ' mini_patch:' + str(mini_patch) + "/" + str() + 'traning time:' + str(time) + '\n'
        # f.write(title)
        info = '## training    epoch:' + str(epoch) + '   mini_patch: ' + str(mini_patch) + "/" + str(total) + '   training_time: ' + str(time) + 's\n'
        f.write(info)
        f.write('\n')
        f.write('## acc = ' + str(acc))
        f.write('\n')
        f.write("***")

        f.close()




class minipatch:
    def __init__(self, clus_center, name):
        self.mini_patch_name = name
        self.clus_center_list = clus_center
        self.clus_num = len(clus_center)

def merge_mini_patch(patches_list, thresholed)-> minipatch:
    """_summary_

    Args:
        patches_dict (_type_): {name:center_fea}
    """
    def merge(buffer, A:int, B:int, thresholed):
        #寻找相似度高的
        temp_list = []
        new_name = len(buffer) + 1 
        a = buffer[A]
        b = buffer[B]
        for i in range(a.clus_num):
            sim_buffer = []#用于存储a[i]和b的所有相似度，万一有多个阈值以上的
            for j, _ in enumerate(b.clus_center_list):#这样写可以让内层循环的长度随b变化而变化
                # similarity = F.pairwise_distance(a.clus_center_list[i], b.clus_center_list[j], p=2)
                similarity = torch.cosine_similarity(a.clus_center_list[i], b.clus_center_list[j])
                sim_buffer.append(similarity)
            print(sim_buffer)
            max_idx = sim_buffer.index(max(sim_buffer))
            #大于阈值就融合#取平均
            if max(sim_buffer) > thresholed:
                avg = (a.clus_center_list[i] + b.clus_center_list[max_idx]) / 2
                temp_list.append(avg)
                b.clus_center_list.pop(max_idx)
            else:
                temp_list.append(a.clus_center_list[i])
        #最后将b中和a没有相似度的也加进去
        temp_list.extend(b.clus_center_list)
        buffer.pop(B)
        buffer.pop(A)
        buffer.append(minipatch(temp_list, new_name))
            
    minipatchbuffer = []
    for idx, clu_centers in enumerate(patches_list):
        minipatchbuffer.append(minipatch(clu_centers, idx))
    idx = 0
    while(len(minipatchbuffer) != 1):
        merge(minipatchbuffer, 0, 1, thresholed)
    return minipatchbuffer[0]



"""
由于每个小patch中的坐标不是连续的，也不是从零开始的，所以需要重新映射
"""


def label_mapping(true_label):
    """
    将任意的label映射到丛零开始，连续的
    """
    num_label = len(set(true_label))
    #create mapping
    lables_true = sorted(set(true_label))
    label_map = {}
    for i in range(num_label):
        label_map[lables_true[i]] = i
    #替换
    res = []
    for i in true_label:
        res.append(label_map[i])
    return res, label_map


def evaluate(label_, pred):

    # pred = pred.cpu().numpy()
    label, _ = label_mapping(label_)
    nmi = metrics.normalized_mutual_info_score(label, pred)
    ari = metrics.adjusted_rand_score(label, pred)
    f = metrics.fowlkes_mallows_score(label, pred)
    pred_adjusted = get_y_preds(label, pred, len(set(label)))
    # print(f"pre_label:{pred_adjusted[0:10]}")
    # print(f"TRUE_LABEL:{label[0:10]}")
    # print(f"my_predict acc={(pred_adjusted == label).sum() / label.shape[0]}")
    acc = metrics.accuracy_score(pred_adjusted, label)
    f1 = f1_score(label, pred, average = 'weighted')
    return nmi, ari, f, acc, f1


def calculate_cost_matrix(C, n_clusters):
    cost_matrix = np.zeros((n_clusters, n_clusters))
    # cost_matrix[i,j] will be the cost of assigning cluster i to label j
    for j in range(n_clusters):
        s = np.sum(C[:, j])  # number of examples in cluster i
        for i in range(n_clusters):
            t = C[i, j]
            cost_matrix[j, i] = s - t
    return cost_matrix


def get_cluster_labels_from_indices(indices):
    n_clusters = len(indices)
    cluster_labels = np.zeros(n_clusters)
    for i in range(n_clusters):
        cluster_labels[i] = indices[i][1]
    return cluster_labels


def get_y_preds(y_true, cluster_assignments, n_clusters):
    """
    Computes the predicted labels, where label assignments now
    correspond to the actual labels in y_true (as estimated by Munkres)
    cluster_assignments:    array of labels, outputted by kmeans
    y_true:                 true labels
    n_clusters:             number of clusters in the dataset
    returns:    a tuple containing the accuracy and confusion matrix,
                in that order
    """
    # print(cluster_assignments)
    # print(f"counter{Counter(y_true)}")
    confusion_matrix = metrics.confusion_matrix(y_true, cluster_assignments, labels=None)
    # compute accuracy based on optimal 1:1 assignment of clusters to labels
    cost_matrix = calculate_cost_matrix(confusion_matrix, n_clusters)
    # print(f"代价矩阵{cost_matrix}")
    indices = Munkres().compute(cost_matrix)
    # print(f"对照{indices}")
    kmeans_to_true_cluster_labels = get_cluster_labels_from_indices(indices)
    if np.min(cluster_assignments) != 0:
        cluster_assignments = cluster_assignments - np.min(cluster_assignments)

    # print(f"试试{kmeans_to_true_cluster_labels}")
    y_pred = kmeans_to_true_cluster_labels[cluster_assignments]
    return y_pred




def compute_acc(true_label, pre_label, fold_dic, args):
    # 解折叠
    print("start eval")
    pre_label_list = []   
    # print(f"真实标签{Counter(true_label)}")
    for i in pre_label:
        pre_label_list.append(i.item())
    # print(f"前面{Counter(pre_label_list)}")
    while(len(pre_label_list) != len(true_label)):
        #倒着来
        fold_idx = len(pre_label_list) - 1
        fold_nodes = fold_dic.fold_dict[fold_idx]
        fold_label = pre_label_list[fold_idx]
        for i in fold_nodes:
            pre_label_list[i] = fold_label
        pre_label_list.pop(-1)
    nmi, ari, f, acc, f1_score = evaluate(true_label, pre_label_list)
    return acc
    # save_log_eval(args.save_folder_test, big_epoch, wsi_name, acc, mini_patch, time, epoch, total)



@torch.no_grad()
def evaluate_wsi(backbone:torch.nn.Module, gcn: torch.nn.Module,
            wsi_img,
            wsi_dict,
            big_epoch,
            mini_patch,
            total,
            clus_num,
            args
        ):
    backbone.eval()
    gcn.eval()
    print(f"start eval")
    fold_dic = fd(args.batch_size)#用于记录被折叠的点
    stable_dic = sd()#用于记录稳定的点
    wsi_name = wsi_dict[0][0].split("_")[0]
    start = time()
    # print(f"wsi:[{wsi_name}], epoch:[{epoch}]:")
    input_image = wsi_img.to('cuda:0')



    #training
    for i in range(10):
        node_fea = backbone(input_image)
        update_fold_dic(stable_dic, fold_dic)
            #先将折叠中心的node_fea添加
        for k in fold_dic.fold_dict.keys():# 不存在空的折叠点
            node_fea_k = torch.zeros(768).to("cuda:0")
            for node in fold_dic.fold_dict[k]:
                node_fea_k += node_fea[node]
            node_fea_k = node_fea_k / len(fold_dic.fold_dict[k])
            node_fea = torch.cat([node_fea, node_fea_k.unsqueeze(0)], dim = 0)


        node_fea_detach = node_fea.clone().detach()
        g, u_v_pair, edge_fea = new_graph(wsi_dict, fold_dic, node_fea_detach, 1, "cuda:1").init_graph()
        clu_label = Cluster(node_fea=node_fea_detach, cluster_num = clus_num, device='cuda:1').predict()
        #向字典中添加聚类标签#TODO
        for i in range(len(wsi_dict)):
            wsi_dict[i].append(clu_label[i])
        mask_rates = [args.mask_rate_high, args.mask_rate_mid, args.mask_rate_low]#各个被mask的比例
        mask_idx, fea_center, fea_edge, sort_idx_rst, cluster_center_fea = chooseNodeMask(node_fea_detach, clus_num, mask_rates, wsi_dict, "cuda:1", stable_dic, clu_label)#TODO 检查数量
        mask_edge_idx = chooseEdgeMask(u_v_pair, clu_label, sort_idx_rst, {"inter":0.1, "inner":0.1, "random":0.1} )
        node_fea[mask_idx] = 0
        edge_fea[mask_edge_idx] = 0
        print(f"this epoch mask nodes:{len(mask_idx)}, mask edges: {len(mask_edge_idx)}")

        g = g.to("cuda:1")
        edge_fea = edge_fea.to("cuda:1")
        node_fea = node_fea.to("cuda:1")
        predict_nodes = gcn(g, node_fea, edge_fea)
        pys_center, pys_edge = compute_pys_feature(wsi_dict, args.pos_choose) #计算物理特征
        fea2pos(fea_center, fea_edge, pys_center, pys_edge)#统计对齐信息并打印
        # save_log(args.log_folder, wsi_name, epoch, clu_label, mask_idx)
        #将这旧的聚类标签删除
        true_label =[]
        # print(wsi_dict[1])
        for i in range(len(wsi_dict)):
            wsi_dict[i].pop(-1)
            true_label.append(int(wsi_dict[i][3]))
        pre_label = Cluster(node_fea=predict_nodes.detach(), cluster_num = clus_num, device='cuda:1').predict()
        
        print(fold_dic.fold_dict)
        # 解折叠
        pre_label_list = []   
        # print(f"真实标签{Counter(true_label)}")
        for i in pre_label:
            pre_label_list.append(i.item())
        # print(f"前面{Counter(pre_label_list)}")
        while(len(pre_label_list) != len(wsi_dict)):
            #倒着来
            fold_idx = len(pre_label_list) - 1
            fold_nodes = fold_dic.fold_dict[fold_idx]
            fold_label = pre_label_list[fold_idx]
            for i in fold_nodes:
                pre_label_list[i] = fold_label
            pre_label_list.pop(-1)
        # print(f"后面后面后面{Counter(pre_label_list)}")
        nmi, ari, f, acc, f1_score = evaluate(true_label, pre_label_list)
        # except:
        #     print("预测的标签种类少于label的种类，无法评价")
        #     acc = -1
        print(f"acc:{acc}")
    # save_log_eval(args.save_folder_test, big_epoch, wsi_name, acc, mini_patch, time, epoch, total)
