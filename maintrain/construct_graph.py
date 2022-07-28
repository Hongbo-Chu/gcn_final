from ast import arg
from errno import EFAULT
from os import rename
import numpy as np
import torch
from tqdm import tqdm
from collections import Counter
import torch.nn.functional as F
import dgl
from maintrain.utils.utils import Cluster, neighber_type
"""
从backbone来的数据：
1. nodefeature:
    shape[node_num, feature_dim]
2. positionfeature:
    shape:[node_num, [pos_x, pos_y]]
构建：
1. edge_feature
    shape[node_num^2, edge_dim]
    
2. adjacent matric

"""

def neighbor_idx(node_idx, wsi, n):
    """用于找一个点周围邻居的idx
        args:
            node_idx:要找邻居的点的idx
            wsi:  {idx: (name, (x_true, y_true), (x, y))}
            n:找几圈邻居
        returns：
            邻居的idx，列表形式
    """
    pos_dict = {}
    for i in range(len(wsi)):
        pos = wsi[i][2]
        label = wsi[i][1]
        pos_dict[tuple(pos)] = i
    
    neighbor_idx = []
    pos = wsi[node_idx][2]#当前点的坐标
    for i in range(pos[0]-n, pos[0]+n+1):
        for j in range(pos[1]-n, pos[1]+n+1):
            if (i, j) in pos_dict:
                neighbor_idx.append(pos_dict[(i, j)])
    return neighbor_idx

def L2_dist(x, y):
    """
    a fast approach to compute X, Y euclidean distace.
    Args:
        x: pytorch Variable, with shape [m, d]
        y: pytorch Variable, with shape [n, d]
    Returns:
        dist: pytorch Variable, with shape [m, n]
    """
    #y: pytorch Variable, with shape [n, d]
    m, n = x.size(0), y.size(0)
    # xx经过pow()方法对每单个数据进行二次方操作后，在axis=1 方向（横向，就是第一列向最后一列的方向）加和，此时xx的shape为(m, 1)，经过expand()方法，扩展n-1次，此时xx的shape为(m, n)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    # yy会在最后进行转置的操作
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    # torch.addmm(beta=1, input, alpha=1, mat1, mat2, out=None)，这行表示的意思是dist - 2 * x * yT 
    dist.addmm_(x, y.t(), beta=1, alpha=-2)
    # clamp()函数可以限定dist内元素的最大最小范围，dist最后开方，得到样本之间的距离矩阵
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist



# def prepoess_file_list(wsi, cluster_num, device):
#     """根据输入的文件名列表构件数据字典,并为每一个文件创建一个唯一idx

#     Args:
#         wsi (_type_): 格式：{idx: name, (x_t, y_t), (x, y)}
#     returns:
#         {idx: (name, (x, y), (x_true, y_true), label)}
#         node_fea:[N, dim]
#     """
#     true_pos = []
#     wsi_pos = [[int(kk[0].split("_")[3]), int(kk[0].split("_")[4]), kk[0], kk[1]] for kk in wsi]
#     #wsi_pos格式1：[(x_true, y_true), name, node_fea]
#     wsi_min_x = min([x[0] for x in wsi_pos])
#     wsi_min_y = min([x[1] for x in wsi_pos])
#     wsi_pos = [[(x[0]-wsi_min_x) // 512, (x[1]-wsi_min_y) // 512, x[2], x[3], [x[0], x[1]] ] for x in wsi_pos]
#     #wsi_pos格式2：[x, y, name, node_fea，[x_true, y_true]]
#     ww = sorted(wsi_pos, key = lambda element: (element[0], element[1]))
#     ww_dic = {}
#     tensor_list = []
#     for idx, w in enumerate(ww):
#         true_pos.append(torch.tensor(w[4]).unsqueeze(0))
#         ww_dic[idx] = [w[2], (w[0], w[1]), w[3], w[4]]
#         tensor_list.append(torch.tensor(w[3]).unsqueeze(0))
#     #将node-fea按照处理后的顺序变成tensor方便之后使用
#     node_fea_tensor = torch.cat(tensor_list, dim = 0)
#     # print(f"shape of node_fea{node_fea_tensor.size()}")
#     #生成聚类标签
#     # print(args)
    
#     return ww_dic, node_fea_tensor, clu_res, true_pos



class new_graph:
    def __init__(self, wsi, fold_dict, node_fea, edge_enhance, device) -> None:
        """根据node_fea和读取的文件名建图
         Args:
            wsi (_type_): 格式：{idx: name, (x_t, y_t), (x, y)}
            cluster_num: 聚类的种类
            graph_mlp:用于边的特征映射的mlp
        """
        self.device = device
        self.wsi_dic = wsi
        self.fold_dict = fold_dict.fold_dict
        self.node_fea = node_fea
        self.node_num = len(self.node_fea)
#         self.d = self.d.to(self.device)
        self.node_fea = self.node_fea.to(self.device)
        self.edge_enhance = edge_enhance
            
    def init_graph(self):
        e_fea = L2_dist(self.node_fea, self.node_fea)
        # e_fea = self.edge_mlp(e_fea).view(self.node_num, self.node_num)#[n^2, 6] -> [n^2, 1] -> [n, n]
        #将所有不到阈值的edge_fea归零
#         print(e_fea)
        threshold = int(e_fea.mean())
        threshold_e = torch.threshold(e_fea, threshold, 0)#size() = n,n
        print(threshold_e.size()) 
        #然后判断需要增强的邻居节点
        edge_enhance = []
        for node in range(len(self.wsi_dic)):
            temp = torch.zeros(1, self.node_num)
            neighbor_edge = neighbor_idx(node, self.wsi_dic, 1)
            #折叠的点不加强
            neighbor_edge.extend([0 for _ in range(len(list(self.fold_dict.keys())))])
            temp[0][neighbor_edge] = self.edge_enhance
            edge_enhance.append(temp)
        #TODO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!被折叠的边要不要增强
        for ii in range(len(list(self.fold_dict.keys()))):
            edge_enhance.append(torch.zeros(1, self.node_num))
        #一个n x n的tensor其中，只有对应位置的值为1
        edge_enhance = torch.cat(edge_enhance, dim=0).to(self.device)
        # print(f"用于边增强的矩阵的形状{edge_enhance.size()}")
        #现在的边值是根据周围一圈邻居的值和原edge_fea生成的
        threshold_e = (edge_enhance + threshold_e)
        u = []
        v = []
        ee = []
        fold_nodes = [] #所有被折叠的点
        count = 0 #用于计被折叠点出现的位置
        count_list = [] # 用于记录所有的被折叠点的位置
        for i in self.fold_dict.keys():
            fold_nodes.extend(self.fold_dict[i])
            
        for i in tqdm(range(self.node_num)): #全连接图
            for j in range(i+1 ,self.node_num):
                if threshold_e[i][j] != 0 and i != j:#判断在阈值之内可以，且无自环
                    if i in fold_nodes or j in fold_nodes:#记录被折叠点的坐标，因为后面添加的点的连接要根据它都包含了哪些点决定
                        count_list.append(count)#不用记录具体信息，因为反正这些点都要去掉
                    u.append(i)
                    v.append(j)
                    ee.append((threshold_e[i][j]).unsqueeze(0))
                    count += 1
        # for fd in self.fold_dict.keys():
        #     if len(self.fold_dict[fd]) != 0:#代表没有失效
                #要根据所有被折叠的点的连接情况来判断新的折叠中心的连接情况

                # dest = []#存放折叠中心的所有目标点
                # for node in self.fold_dict[fd]:#统计新产生的点要和哪些旧点产生关系
                #     for idx, val in enumerate(u):
                #         if val == node and v[idx] not in self.fold_dict[fd]: #还要去掉内部的点
                #             dest.append(v[idx]) 
                # dest = dict(Counter(dest))
                # for d in dest.keys():
                #     weight = dest[d] #TODO 用相似度算
                #     u.append(fd)
                #     v.append(d)
                #     ee.append(weight)
        #去除掉所有的旧点，按照先后顺序倒着来
        count_list.reverse()
        for dele in count_list:
            ee.pop(dele)
            u.pop(dele)
            v.pop(dele)
        temp_graph = dgl.graph((u, v))
        self.graph = dgl.add_reverse_edges(temp_graph).to(self.device)

        print(self.graph)
        ee = torch.cat(ee, dim=0).unsqueeze(1)#最终的edge_fea，是将那些为0的边都去掉了
        ee = torch.cat([ee,ee], dim =0)
        # print(f"提取的边的特征：{ee.size()}")
        # print(f"建完的图{self.graph}")
        # print(f"edge_fea{ee.size()}")
        return self.graph, (u, v), ee


                
if __name__ == '__main__':
    #模拟输入
    save_dict_path = 'C:/Users/86136/Desktop/new_gcn/wsi_dict.npy'
    wsi_dict = np.load(save_dict_path, allow_pickle='TRUE')
    wsi = wsi_dict.item()['43']
    wsi = [[w, torch.randn(1, 128)] for w in wsi ]
    wsi = wsi
    # a, b = prepoess_file_list(wsi, 6)
    # print(a[0])
    aa = new_graph(wsi, 6)
    aa.init_graph()
                