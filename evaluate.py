# from matplotlib.font_manager import _Weight
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
from maintrain.utils.fold import update_fold_dic
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




def evaluate(label, pred):
    pred = pred.cpu().numpy()
    nmi = metrics.normalized_mutual_info_score(label, pred)
    ari = metrics.adjusted_rand_score(label, pred)
    f = metrics.fowlkes_mallows_score(label, pred)
    pred_adjusted = get_y_preds(label, pred, 9)
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
    confusion_matrix = metrics.confusion_matrix(y_true, cluster_assignments, labels=None)
    # compute accuracy based on optimal 1:1 assignment of clusters to labels
    cost_matrix = calculate_cost_matrix(confusion_matrix, n_clusters)
    indices = Munkres().compute(cost_matrix)
    kmeans_to_true_cluster_labels = get_cluster_labels_from_indices(indices)
    if np.min(cluster_assignments) != 0:
        cluster_assignments = cluster_assignments - np.min(cluster_assignments)
    y_pred = kmeans_to_true_cluster_labels[cluster_assignments]
    return y_pred



@torch.no_grad()
def evaluate_wsi(backbone:torch.nn.Module, gcn: torch.nn.Module,
            graph_mlp:torch.nn.Module,
            wsi_img,
            wsi_dict,
            big_epoch,
            mini_patch,
            total,
            args
        ):
    backbone.eval()
    gcn.eval()
    graph_mlp.eval()
    print(f"start eval")
    fold_dic = fd(args.batch_size)#用于记录被折叠的点
    stable_dic = sd()#用于记录稳定的点
    wsi_name = wsi_dict[0][0].split("_")[0]
    for epoch in range(args.epoch_per_wsi):
        start = time()
        # print(f"wsi:[{wsi_name}], epoch:[{epoch}]:")
        input_image = wsi_img.to('cuda:0')



        #training
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
        g, u_v_pair, edge_fea = new_graph(wsi_dict, 9, graph_mlp, fold_dic, node_fea_detach, "cuda:1").init_graph()
        clu_label = Cluster(node_fea=node_fea_detach, cluster_num = 9, device='cuda:1').predict()
        #向字典中添加聚类标签#TODO
        for i in range(len(wsi_dict)):
            wsi_dict[i].append(clu_label[i])
        mask_rates = [args.mask_rate_high, args.mask_rate_mid, args.mask_rate_low]#各个被mask的比例
        mask_idx, fea_center, fea_edge, sort_idx_rst, cluster_center_fea = chooseNodeMask(node_fea_detach, args.cluster_num, mask_rates, wsi_dict, "cuda:1", stable_dic)#TODO 检查数量
        mask_edge_idx = chooseEdgeMask(u_v_pair, clu_label,sort_idx_rst, {"inter":0.1, "inner":0.1, "random":0.1} )
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
        print(wsi_dict[1])
        for i in range(len(wsi_dict)):
            wsi_dict[i].pop(-1)
            true_label.append(int(wsi_dict[i][3]))
        pre_label = Cluster(node_fea=predict_nodes.detach(), cluster_num = args.cluster_num, device='cuda:1').predict()
        nmi, ari, f, acc, f1_score = evaluate(true_label, pre_label)
        print(f"acc:{acc}")
        save_log_eval(args.save_folder_test, big_epoch, wsi_name, acc, mini_patch, time, epoch, total)
