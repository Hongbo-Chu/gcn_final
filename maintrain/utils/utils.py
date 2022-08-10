import enum
from platform import node
from tkinter import BROWSE
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score 
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import numpy as np
from torch import tensor
import torch
import time
from collections import Counter
import torch.nn.functional as F
import random
from tqdm import tqdm
from kmeans_pytorch import kmeans
from sklearn.cluster import KMeans
import copy
from matplotlib import pyplot as plt
class Cluster:
    """
    based on sklearn: https://scikit-learn.org/stable/modules/clustering.html
    v2.0:参考refine的那片，使用聚类+二分，通过sc——score来判断是否继续二分下去
    """

    def __init__(self, node_fea, device, method = "K-means") -> None:
        """
        inputs: node_features
        
        output: label of each node
            shape: 1 x N (a list of cluster)
        params:
        node_fea: input noode feature, shape -> num_nodes x feature_dims
        method: cluster methods
        cluster_num: number of clusters
        **kwargs: oarameters of spatracl cluster
        returns:
            predict返回tensor(1, N)用于表示每个点的类别。
        """
        
        # our support cluster methods
        self.clusters= {'K-means':self.K_means, 'spectral':self.spectral, 'Affinity':self.Affinity, 'hierarchy': self.hierarchy}
        assert method in self.clusters.keys(), "only support K-means, spectral, Affinity, hierarchy"
        # self.cluster_num = cluster_num
        self.methods = method
        self.node_fea = node_fea
        self.device =device     

    def bi_partition(self, cluster_node_fea, cluster_label, target, **kwargs):# TODO 方法可选
        """
        args:
            cluster_node_fea: 将node_fea按照cluster分开[[],[],[],...]
            cluster_label: 与上面node_fea对应的idx[[],[],[],...]
            target: cluster中要被二分的那个类
            method: 用于二分的聚类方法
        returns:
            将那个被拆分的类放回所有的类当中再返回
            node_fea_clus: [[],[],[],...]
            idx_clus: [[],[],[],...]
        """
        if len(cluster_node_fea[target]) == 0:
            return cluster_node_fea
        clusters_copy = copy.deepcopy(cluster_node_fea)
        label_copy = copy.deepcopy(cluster_label)
        clus_bi_copy = copy.deepcopy(cluster_node_fea[target])
        label_bi_copy = copy.deepcopy(cluster_label[target])
        pre_label =self.clusters[self.method](kwargs, x = clus_bi_copy)
        #将
        
        bi_paritition_0 = []
        bi_paritition_1 = []    
        bi_label_0 = []
        bi_label_1 = []
        for k, val in enumerate(pre_label):
            if val == 0:
                bi_paritition_0.append(clus_bi_copy[k])
                bi_label_0.append(label_bi_copy[k])
                # bi_partition_0_true_label.append(input_label[k])#真实标签，最后验证用
            else:
                bi_paritition_1.append(clus_bi_copy[k]) 
                bi_label_1.append(label_bi_copy[k])
                # bi_partition_1_true_label.append(input_label[k])#真实标签，最后验证用
        clusters_copy.pop(target)
        clusters_copy.append(bi_paritition_0)
        clusters_copy.append(bi_paritition_1)
        label_copy.pop(target)
        label_copy.append(bi_label_0)
        label_copy.append(bi_label_1)
        return clusters_copy, label_copy
 
    def predict(self, **kwargs):
        """
        v2.0 不断的二分
        returns:
            labels:以列表的形式按顺序返回
        """
        node_fea_clus = [[i for i in self.node_fea]] #改成列表嵌套的形式
        label_clus = [[i for i in range(len(self.node_fea))]] # 初始化标签都为0
        final_label = [-1] * len(self.node_fea)
        # clus_method = self.clusters[self.method]()
        sc_best = sc_pre = -1000
        i_best = 0
        while(i_best != -1):
            i_best = -1
            i = 0
            while(i < len(node_fea_clus)):# 挑选出最适合二分的那个cluster
                _, idx_clus = self.bi_partition(node_fea_clus, label_clus, i)
                #将[[],[],[],...]形式的label转为[]形式
                pre_label = [-1] * len(self.node_fea)
                for label, clu_i in enumerate(idx_clus):
                    for itm in clu_i:
                        pre_label[itm] = label
                assert -1 not in pre_label, "对不上"
                sc_new = silhouette_score(self.node_fea.cpu().numpy(), pre_label, metric='cosine') # (node_fea, pre_label)
                if sc_new > sc_best:
                    sc_best = sc_new
                    i_best = i
                i += 1
            if i_best != -1: #将上面找出的i_best拆分
                node_fea_clus, label_clus = self.bi_partition(node_fea_clus, label_clus, i_best)
                final_label = [-1] * len(self.node_fea)
                for label, clu_i in enumerate(label_clus):
                    for itm in clu_i:
                        final_label[itm] = label
                assert -1 not in final_label, "对不上"
                sc_pre = silhouette_score(self.node_fea.cpu().numpy(), final_label, metric='cosine') # (node_fea, pre_label)
                sc_best = sc_pre
            else: #结束二分
                break
        # result = self.clusters[self.methods](**kwargs)
        clus_num = len(set(final_label))
        print(f"最终分为{clus_num}类")
        return final_label, clus_num
        
    def K_means(self, x, **kwargs):
#         print("use cluster-method: K-means")
#         self.model = KMeans(self.cluster_num)
#         pre_label = self.model.fit_predict(self.node_fea.cpu())
# #         print(pre_label)
#         return pre_label
        print("use cluster-method: K-means")
        print(type(x))
        print(len(x))
        pre_label, cluster_centers = kmeans(torch.stack(x), num_clusters=2, distance='euclidean', device=torch.device('cuda:0'))
        pre_label = pre_label.to('cuda:0')
        return pre_label

    def hierarchy(self, **kwargs):
        threshold_dis = kwargs['threshold_dis']
        if 'method_h' not in kwargs.keys():
            method_h = 'ward'
        else:
            method_h = kwargs['method_h']
        self.node_fea = self.node_fea.to("cpu")
        Z = linkage(self.node_fea, method_h)
        q = sorted(Z[:,2])
        thres = q[int(len(q)*0.99)]
        pre_label = fcluster(Z, thres, criterion='distance')
        print(Z[:,2].mean())
        print(f"sss{len(set(pre_label))}")
        # plt.figure(figsize=(50, 10))
        # plt.title('Hierarchical Clustering Dendrogram')
        # plt.xlabel('sample index')
        # plt.ylabel('distance')
        # dendrogram(Z, leaf_rotation=90., leaf_font_size=8.)
        # f = plt.gcf()  #获取当前图像
        # f.savefig(r'bb.png')
        assert False
        pre_label = fcluster(Z, threshold_dis, criterion='distance')
        # print(pre_label)
        clus_num = len(set (pre_label))
        # print(clus_num)
        #映射到从零开始
        for i in range(len(pre_label)):
            pre_label[i] = pre_label[i] - 1
        return pre_label, clus_num
    
    def spectral(self, x, **kwargs):
        Scluster = SpectralClustering(n_clusters=2, affinity='nearest_neighbors',n_neighbors=10)# TODO 补充谱聚类参数
        
        return  Scluster.fit_predict(self.node_fea)# TODO 验证输入格式
    
    def  Affinity(self):
        pass
    

def euclidean_dist(x, center):
    """
    a fast approach to compute X, Y euclidean distace.
    Args:
        x: pytorch Variable, with shape [m, d]
        center: pytorch Variable, with shape [1, d]
    Returns:
        dist: pytorch Variable, with shape [1, m]
    """
    temp = [center for _ in range(x.size(0))]
    y = torch.cat(temp, dim=0)
    #y: pytorch Variable, with shape [n, d]
    m, n = x.size(0), y.size(0)
    # xx经过pow()方法对每单个数据进行二次方操作后，在axis=1 方向（横向，就是第一列向最后一列的方向）加和，此时xx的shape为(m, 1)，经过expand()方法，扩展n-1次，此时xx的shape为(m, n)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    # yy会在最后进行转置的操作
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    # torch.addmm(beta=1, input, alpha=1, mat1, mat2, out=None)，这行表示的意思是dist - 2 * x * yT 
    dist.addmm_(x, y.t(), beta = 1, alpha = -2)
    # clamp()函数可以限定dist内元素的最大最小范围，dist最后开方，得到样本之间的距离矩阵
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    res = []
    for i in range(m):
        res.append(dist[i][0]) 
    return res

def samesort(list1, list2):
    """
    用list1的排序方法排序list2
    returns:
        list1, list2
    """
    return zip(*sorted(zip(list1, list2))) 

def compute_clus_center(node_fea, clus_res):
    node_fea_list = [[] for _ in range(len(set(list(clus_res))))]
    for idx, n in enumerate(node_fea):
        node_fea_list[clus_res[idx]].append(n)
    for idx in range(len(node_fea_list)):
        node_fea_list[idx] = torch.stack(node_fea_list[idx]).mean()
    return node_fea_list

    

def split2clusters(node_fea, cluster_num, cluster_res, device, cluster_method = "K-means"):
    """
    split nodes into different clusters
    将node_fea按照聚类的结果进行分类，
    args:
        node_fea: tensor(n,fea_dim)的形式
        cluster_num：聚类种类
        device：聚类使用的设备
    returns:
        node_fea_list：按照聚类的种类分的列表[[tensor(1,node_fea)， tensor(1,node_fea)....],[],[]....],
        node_idx_list:按照聚类种类存储的每一类中的node_indx（具有唯一标识性），[[int, int,int,......],[],[]...]
    """
    
    node_fea_list = [[] for _ in range(cluster_num)] # 用于存放每一类cluster的node_fea(tensor)
    node_idx_list = [[] for _ in range(cluster_num)] # 用于存放每一类cluster的标签(int)
    # cluster_res = Cluster(node_fea=node_fea, cluster_num=cluster_num, method=cluster_method, device=device).predict()
    #按照聚类标签分类
    for idx, clu in enumerate(cluster_res):
        try:
            node_fea_list[clu].append(node_fea[idx].unsqueeze(0))
            node_idx_list[clu].append(idx)
        except:
            print(f"啥超了啊{idx} {clu} {len(node_fea_list)}")
    return node_fea_list, node_idx_list

def chooseNodeMask(node_fea, cluster_num, mask_rate:list, wsi, device, stable_dic, cluster_res, cluster_method = "K-means"):
    """
    choose which nodes to mask of a certain cluster
    args:
        maskrate: list,用于存放三种相似度的mask比例。[高，中，低]
        wsi:建图的时候返回的结构体，用于在加mask前判断是否重合
    return: 
        被masknode的idx，list of nodes_idx
        与聚类中心高相似度的点 list of nodes_idx
        与聚类中心低相似度的点 list of nodes_idx
        cluster_center_list:所有类的聚类中心 list of nodes_idx
        sort_idx_rst:每一类节点的聚类的排序结果，list of list形式
    v2.0:
    添加了对于折叠点的选取，就是将重合的那些点添加到stabel_dict当中去，然后维护这个字典，
    一定轮数之后判断是否折叠
    """
    mask_node_idx = [] # 用于存储最终mask点的idx
    high = [] # 用于存储高相似度
    low = [] #用于存储低相似度
    node_fea_list, node_idx_list = split2clusters(node_fea, cluster_num, cluster_res, device, cluster_method)
    sort_idx_rst = [[] for i in range(cluster_num)]#用于存放每一类按照相似度从大到小的排序结果，后面edgemask的时候要用。
    cluster_center_list = []
    #取mask前先要判断是否重合
    pys_center, pys_edge = compute_pys_feature(wsi=wsi, n = 1)#计算处于物理中心和边缘的点
    # print(f"边缘的点和中心的点{len(pys_edge)} {len(pys_center)}")
    #对每一类点分别取mask

    for idx, (feats, idxs) in enumerate(zip(node_fea_list, node_idx_list)):
        #feats的格式是[tensor,tessor....],先要拼成一个tensor
        feats = torch.cat(feats, dim = 0)
        # print(f"feat{feats.size()}")
        cluster_center = feats.mean(dim=0)
        cluster_center_list.append(cluster_center)
        #计算任一点和中心的欧氏距离
        # print(f"center:{cluster_center.size()}")
        dist = euclidean_dist(feats, cluster_center.unsqueeze(0))
        #按照特征与中心特征的相似度，对node_idx进行排序
        sorted_disrt, sorted_idex = samesort(dist, idxs)
        sort_idx_rst[idx].extend(sorted_idex)
        #计算聚类半径，由聚类中心减去和他相似度最低的元素
        #对于index取不同位置的点进行mask
        
        for i, rate in enumerate(mask_rate):
            mask_num = int(len(sorted_idex) * rate)
            if i == 0:#高相似度
                #先判断是否重合
                nodes_tobe_mask = sorted_idex[:mask_num]
                #通过差集求取
                mask_nodes_set = set(nodes_tobe_mask) - set(pys_center)
                mask_node_idx.extend(mask_nodes_set)
                high.extend(sorted_idex[:mask_num])#概率高的点，但是不一定被加mask
                #直接添加
                # print(f"调试调试{len(pys_center)}")
                stable_dic.add_stable_idx(nodes_tobe_mask, pys_center, idx)
            elif i == 2:#地相似度
                #先判断是否重合
                nodes_tobe_mask = sorted_idex[-mask_num:]
                #通过差集求取
                mask_nodes_set = set(nodes_tobe_mask) - set(pys_edge)
                mask_node_idx.extend(mask_nodes_set)
                low.extend(sorted_idex[-mask_num:])
            else: # 中相似度
                mid = len(sorted_idex) // 2
                mid_pre = mid - (mask_num) // 2
                mask_node_idx.extend(sorted_idex[mid_pre:mid_pre + mask_num])
    return mask_node_idx, high, low, sort_idx_rst, cluster_center_list

def chooseEdgeMask(u_v_pair, clus_label, sort_idx_rst, rates:dict):
    """
    按照策略，选类内，类间和随机三类
    args：
        源节点和目标节点。
        sort_idx_rst:每一类中的indx按照相似度从小到大的顺序排序，format:[[],[],....]
        rates:各种mask的比例共四类，类间，类内半径，类内中心，类内随机。用字典的形式传入。
    return:
        源节点-目标节点对在所有边中的位置，list形式
    """
    u, v = u_v_pair #u,v分别为两个长列表
    pairs_dict = {(u[i], v[i]): i for i in range(len(u))}#将u,v变成list of pair的形式，方便后面查找
    diff = []
    same = [[] for _ in range(len(set(clus_label)))]
    # same = []
    mask_edge_pair = [] # 最终要被masked边 = 类内 + 类间
    #先调出目标和源同属一类的
    # print(f"hhh{len(u)},{len(v)}{len(clus_label)}")
    for i in range(len(u)):
        if clus_label[u[i]] != clus_label[v[i]]:
            diff.append((u[i], v[i]))
        else:
            same[clus_label[u[i]]].append((u[i], v[i]))
    # 类间
    random.shuffle(diff)
    mask_edge_pair.extend(diff[:int(len(u) * rates['inter'])])
    #类内半径
    def judge_rad(u, v, sort_idx, rate):
        """ 
        寻找符合条件的类内半径
        """
        top = sort_idx[:int(len(sort_idx) * rate)]
        last = sort_idx[-int(len(sort_idx) * rate):]
        return True if ((u in top) and (v in last)) or((u in last) and (v in top)) else False

    def judge_center(u, v, sort_idx, rate):
        top = sort_idx[:int(len(sort_idx) * rate)]
        return True if (u in top) and (v in top) else False

    temp = []
    for idx, clus in enumerate(same):
        #如果一个在前10%,一个在后10%就算类内半径
        for u_v in clus:
            u_ = u_v[0]
            v_ = u_v[1]
            if judge_rad(u_, v_, sort_idx_rst[idx], 0.1):
                temp.append(u_v)
            if judge_center(u_, v_, sort_idx_rst[idx], 0.1):
                temp.append(u_v)
    random.shuffle(temp)
    mask_edge_pair.extend(temp[:int(len(u) * rates['inner'])])
    
    #随机
    temp2 = []#将same打散，用于取随机
    for i in same:
        temp2.append(i)
    #去重
    random.shuffle(temp2)
    count = len(u) * rates['random']
    for i in range(len(temp2)):
        if count == 0:
            break
        if temp2[i] not in temp:
            mask_edge_pair.extend(temp2[i])
            count -= 1
    
    #最后要将edge_pair的数据转化为edge_idx
    edge_idx = []
    print(f"choose edge_mask")
    print(len(mask_edge_pair))
    for i, pair in tqdm(enumerate(mask_edge_pair)):
        # print(i)
        idx = pairs_dict[pair]
        edge_idx.append(idx)

    return edge_idx
def neighber_type(pos, n, pos_dict):
    """查找周围n圈邻居的标签

    Args:
        pos (tuple(x, y)): 点的坐标
        n (int): 几圈邻居
        pos_dict: 用于存储所有点信息的字典{(x, y): label}
    returns:
        1. 所有值，以及数量，字典的形式{val: num}
        2. 邻居标签的种类 int
        3.邻居的标签
    """
    neighbers = []
    for i in range(pos[0]-n, pos[0]+n+1):
        for j in range(pos[1]-n, pos[1]+n+1):
            if (i, j) in pos_dict:
                neighbers.append(pos_dict[(i,j)])
    return Counter(neighbers), len(list(Counter(neighbers).keys()))



def compute_pys_feature(wsi, n):
    """寻找物理特征上中心和边缘的点

    Args:
        wsi (_type_): 建图时候使用的wsi结构体， {idx: (name, (x, y), ndoe_fea, (x_true, y_true), label)}
        n:查找周围n全邻居
    returns:
        center_nodes：位于中心的点,list of nodes的形式.
        edge_nodes: 位于边缘的点,list of nodes的形式.
    """
    center_nodes = []
    edge_nodes = []
    pos_dict = {}
    for i in range(len(wsi)):
        pos = wsi[i][2]
        label = wsi[i][-1]
        pos_dict[tuple(pos)] = label
    # print(pos_dict)
    for j in range(len(wsi)):
        pos = wsi[j][2]
        _, num_type = neighber_type(pos, n, pos_dict)
        if num_type == 1:
            center_nodes.append(j)
        else:
            edge_nodes.append(j)
    # print(f"center&edge{len(center_nodes)}， {len(edge_nodes)}")
    return center_nodes, edge_nodes


def fea2pos(center_fea, edge_fea, center_pos, edge_pos):
    """判断特征空间和物理空间的对应点是否对齐
    """
    print(f"特征空间中心点共{len(center_fea)}个，特征空间边缘点共{len(edge_fea)}个 \n 物理空间中心点共{len(center_pos)}个，物理空间边缘点共{len(edge_pos)}个")
    #找中心点对齐点
    center_map = set(center_fea).intersection(center_pos)
    print(f"中心对齐的点有{len(center_map)}个")
    #找边缘对齐点
    edge_map = set(edge_fea).intersection(edge_pos)
    print(f"边缘对齐的点有{len(edge_map)}个")


class minipatch:
    def __init__(self, clus_center_dict, name):
        """
        clus_center_dict:{clu_center:[truelabel]}
        """
        self.mini_patch_name = name
        self.clus_center_list = []
        self.clus_truelabel = [[] for _ in range(len(clus_center_dict))]
        for idx, k in enumerate(clus_center_dict.keys()):
            self.clus_center_list.append(k)
            self.clus_truelabel[idx].extend(clus_center_dict[k])
        self.clus_num = len(clus_center_dict)

def merge_mini_patch(patches_list, thresholed)-> minipatch:
    """_summary_

    Args:
        patches_dict (_type_): {name:center_fea}
    """
    def merge(buffer, A:int, B:int, thresholed):
        #寻找相似度高的
        temp_center_list = []
        temp_label_list = []
        new_name = buffer[-1].mini_patch_name + 1
        a = buffer[A]
        b = buffer[B]
        for i in range(a.clus_num):
            sim_buffer = []#用于存储a[i]和b的所有相似度，万一有多个阈值以上的
            for j, _ in enumerate(b.clus_center_list):#这样写可以让内层循环的长度随b变化而变化
                # similarity = F.pairwise_distance(a.clus_center_list[i], b.clus_center_list[j], p=2)
                similarity = torch.cosine_similarity(a.clus_center_list[i].unsqueeze(0), b.clus_center_list[j].unsqueeze(0))
                sim_buffer.append(similarity)
            max_idx = sim_buffer.index(max(sim_buffer))
            # 大于阈值就融合#取平均
            if max(sim_buffer) > thresholed:
                avg = (a.clus_center_list[i] + b.clus_center_list[max_idx]) / 2
                temp_center_list.append(avg)
                qq = []
                qq.extend(a.clus_truelabel[i])
                qq.extend(b.clus_truelabel[max_idx])
                temp_label_list.append(qq)
                b.clus_center_list.pop(max_idx)
                b.clus_truelabel.pop(max_idx)
            else:
                temp_center_list.append(a.clus_center_list[i])
                temp_label_list.append(a.clus_truelabel[i])
        #最后将b中和a没有相似度的也加进去
        temp_center_list.extend(b.clus_center_list)
        for i in b.clus_truelabel:
            temp_label_list.append(i)
        buffer.pop(B)
        buffer.pop(A)
        res = {}
        for idx, center in enumerate(temp_center_list):
            res[center] = temp_label_list[idx]
        buffer.append(minipatch(res, new_name))
            
    minipatchbuffer = []
    for idx, clu_centers in enumerate(patches_list):
        minipatchbuffer.append(minipatch(clu_centers, idx))
    idx = 0
    while(len(minipatchbuffer) != 1):
        merge(minipatchbuffer, 0, 1, thresholed)
    return minipatchbuffer[0]


def evaluate(x:minipatch):
    labels = x.clus_truelabel
    total_num = 0
    true_pred = 0
    for lb in labels:
        Count_lb = Counter(lb)
        pre_lb = list(Count_lb.keys())[0]
        true_pred += Count_lb[pre_lb]
        total_num += len(lb)
    acc = true_pred / total_num
    return acc

# if __name__  == '__main__':
#     """
#     模拟backbone的输入： 600 * 128
#     """
#     node_fea = torch.randn(3000, 128)
#     a = chooseNodeMask(node_fea=node_fea,cluster_num=6, mask_rate=[0.1, 0.1, 0.1])
#     print(len(a))
    
#     import numpy as np
#     from collections import Counter
#     wsi_dict = np.load(save_dict_path, allow_pickle='TRUE')
#     wsi = wsi_dict.item()['43']
#     wsi_pos = [[int(kk.split("_")[3]), int(kk.split("_")[4]), kk] for kk in wsi]
#     wsi_min_x = min([x[0] for x in wsi_pos])
#     wsi_min_y = min([x[1] for x in wsi_pos])
#     wsi_pos = [[(x[0]-wsi_min_x) // 512, (x[1]-wsi_min_y) // 512, x[2], ] for x in wsi_pos]
#     ww = sorted(wsi_pos, key = lambda element: (element[0], element[1]))
#     ww = {(w[0],w[1]): (w[2], idx, int(np.random.randint(0,5))) for idx, w in enumerate(ww)}
    



# def inner_cluster_loss(node_fea, clu_label, center_fea, mask_nodes, mask_weight):
#     """用于计算类内loss
#         对于更新后的node_fea(N, dim)，生成对应的中心向量矩阵。
#     Args:
#         node_fea (tensor): 更新后的node_fea，require_grade=True
#         clu_label (_type_): 每个点的聚类标签
#         center_fea (_type_): 几个聚类中心的向量,list of tensor的形式
#         mask_nodes:加了mask的点
#         mask_weight:对于mask点的聚类权重
#     """
    
#     optim_matrix = []#由各种中心向量组成，是优化的目标
#     for i in range(len(clu_label)):
#         # if i in mask_nodes:
#         #     optim_matrix.append((1+mask_weight) * center_fea[clu_label[i]])
#         optim_matrix.append(center_fea[clu_label[i]])
#     optim_matrix = torch.cat(optim_matrix, dim = 1) 
#     loss = node_fea.view(1, -1) @ optim_matrix.transpose(1, 0)
    
#     return loss

