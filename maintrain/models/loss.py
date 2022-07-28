from concurrent.futures import thread
from platform import node
from turtle import forward
import torch
import torch.nn.functional as F
import numpy as np

class myloss(torch.nn.Module):
    def __init__(self):
        super(myloss, self).__init__()

    def inner_cluster_loss(self, node_fea, clu_label, center_fea, mask_nodes, mask_weight):
        """用于计算类内loss
            对于更新后的node_fea(N, dim)，分别计算每个node_fea和聚类中心的L2距离
        Args:
            node_fea (tensor): 更新后的node_fea，require_grade=True
            clu_label (_type_): 每个点的聚类标签
            center_fea (_type_): 几个聚类中心的向量,list of tensor的形式
            mask_nodes:加了mask的点
            mask_weight:对于mask点的聚类权重
        """
        #TODO用矩阵的方式优化
        L2_dist = 0
        # print(f"loss检测{node_fea.size()}, {len(center_fea)}, {len(clu_label)}")
        center_fea = torch.stack(center_fea).to("cuda:1")
        for i in range(len(node_fea)):
            L2_dist += F.pairwise_distance(node_fea[i].unsqueeze(0), center_fea[clu_label[i]].unsqueeze(0), p=2)
            if  i in mask_nodes:
                L2_dist += (1 + mask_weight) * F.pairwise_distance(node_fea[i].unsqueeze(0), center_fea[clu_label[i]].unsqueeze(0), p=2)
        return L2_dist

    def inter_cluster_loss(self, node_fea, sort_idx_rst, center_fea, mask_nodes, mask_weight, angle_threshold):
        """类间loss

        Args:
            node_fea (_type_): 节点特征
            clu_label (_type_): 不需要了现在，都包含在sort_idx_rst中了。
            center_fea (_type_): 每一类的聚类中心
            sort_idx_rst (_type_):每类的相似度排序结果[[],[],[]....]
            mask_nodes (_type_): 加了mask的点
            mask_weight (_type_): 对于mask点的聚类权重
            center_fea: 每一类的聚类中心list of tensor, [dim]
        """
        final_loss = 0
        #遍历所有cluster，两两算
        L2_dist = 0
        # print(f"服了{len(center_fea)}")
        for i in range(len(center_fea)):
            for j in range(i+1, len(center_fea)):
                #先找出两个球之间的连线的那个向量
                try:
                    t = (center_fea[i] - center_fea[j]).unsqueeze(0)
                except:
                    print(i)
                    print(j)
                    
                t = t.to("cuda:1")
                #找出边缘点，定义后10%的点为边缘点

                edgenode_i_idx = sort_idx_rst[i][-int((len(sort_idx_rst[i])) * 0.1):]
                edgenode_j_idx = sort_idx_rst[j][-int((len(sort_idx_rst[j])) * 0.1):]
                edgenode_fea_i = [node_fea[i] for i in edgenode_i_idx]
                edgenode_fea_j = [node_fea[j] for j in edgenode_j_idx]
                #计算与连线之间的cos,这里使用torch的余弦相似度函数进行计算
                cos_i = [F.cosine_similarity(t, k.unsqueeze(0)) for k in edgenode_fea_i]
                cos_j = [F.cosine_similarity(t, w.unsqueeze(0)) for w in edgenode_fea_j]
                #然后挑选符合要求的点
                #镜像选择，i选择threshold ~ 1， j选择 (-1)~(-threshold)
                final_i = []
                final_j = []
                sorted_cos_i = sorted(cos_i)
                th_i = sorted_cos_i[int(len(sorted_cos_i) * 0.1)]
                sorted_cos_j = sorted(cos_j)
                th_j = sorted_cos_j[int(len(sorted_cos_j) * 0.1)]
                for k, cosval in enumerate(cos_i):
                    if cosval > th_i:
                        final_i.append(edgenode_fea_i[k].unsqueeze(0))
                for q, cosval in enumerate(cos_j):
                    if cosval > th_j:
                        final_j.append(edgenode_fea_j[q].unsqueeze(0))
                if len(final_i) != 0 and len(final_j) != 0:
                    final_i = torch.cat(final_i, dim=0).mean(dim=0)
                    final_j = torch.cat(final_j, dim=0).mean(dim=0)
                    L2_dist += F.pairwise_distance(final_i.unsqueeze(0), final_j.unsqueeze(0), p=2)
        
        return -L2_dist
                

    def forward(self, node_fea, clu_label, center_fea, mask_nodes, mask_weight, sort_idx_rst):
        return self.inner_cluster_loss(node_fea, clu_label, center_fea, mask_nodes, mask_weight) + self.inter_cluster_loss(node_fea, sort_idx_rst, center_fea, mask_nodes, mask_weight, 0.8)





if __name__ == "__main__":
    pass