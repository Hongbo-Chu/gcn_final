import torch
import argparse
import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import tqdm

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
# from maintrain.models.loss import myloss


wsi_img = torch.load('../4_HE.pt')[:300]
wsi_dict =  dict(np.load('../4_HE.npy', allow_pickle='TRUE').item())
k = {}
for i in range(300):
    k[i] = wsi_dict[i]

input_img = wsi_img[:300].to("cuda:0")
fold_dic = fd(300)#用于记录被折叠的点
stable_dic = sd(1)
backboneModel = build_model('vit').to("cuda:0")
graph_model = GCN(in_dim=768, num_hidden=768, out_dim=768, num_layers=2, dropout=0,activation="prelu", residual=True,norm=nn.LayerNorm).to('cuda:1')
graph_mlp = g_mlp(in_dim=6, hid_dim=16, out_dim = 1).to("cuda:1")
myloss = ml().to("cuda:1")

node_fea = backboneModel(input_img)

node_fea_detach = node_fea.clone().detach()
g, u_v_pair, edge_fea = new_graph(k, 9, graph_mlp, fold_dic, node_fea_detach, "cuda:1").init_graph(threshold=1)
clu_label = Cluster(node_fea=node_fea_detach, cluster_num = 9, device='cuda:1').predict()
for i in range(len(k)):
    k[i].append(clu_label[i])
print("下面开始加mask")
mask_rates = [0.1, 0.1, 0.1]#各个被mask的比例
print("1")
mask_idx, fea_center, fea_edge, sort_idx_rst, cluster_center_fea = chooseNodeMask(node_fea_detach, 9, mask_rates, k, "cuda:1", stable_dic)#TODO 检查数量
print("2")
mask_edge_idx = chooseEdgeMask(u_v_pair, clu_label,[], {"inter":0.1} )
#添加mask
print("3")
node_fea[mask_idx] = 0
edge_fea[mask_edge_idx] = 0

g = g.to("cuda:1")
edge_fea = edge_fea.to("cuda:1")
node_fea = node_fea.to("cuda:1")
predict_nodes = graph_model(g, node_fea, edge_fea)
center_clus_fea = torch.randn(9,768)
loss = myloss(predict_nodes, clu_label, center_clus_fea, [], 0.1, sort_idx_rst)
print(loss)
loss.backward()