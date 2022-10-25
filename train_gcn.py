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


def gcn_train_one_epoch(backbone: torch.nn.Module,
                        gcn: torch.nn.Module,
                        criterion:torch.nn.Module,
                        wsi_dict,
                        mini_patch,
                        args=None):
    gcn.train()
    backbone.eval()
    node_fea = backbone(input_img)
    for epoch in range(args.epoch_per_wsi):
         g, u_v_pair, edge_fea = new_graph(wsi_dict, stable_dic, node_fea_detach, args.edge_enhance, graph_mlp, args.device1).init_graph(args)
