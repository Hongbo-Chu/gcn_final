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

from trainengine_single import save_log
# from maintrain.utils.fold import update_fold_dic



def preprocess(backbone: torch.nn.Module, save_folder,
                    wsi_img,
                    wsi_dict,
                    mini_patch,
                    args=None):
            
    backbone.eval()
    print(wsi_dict[0])
    wsi_name = str(wsi_dict[0][0]).split("_")[0]
    for epoch in range(args.epoch_per_wsi):
        debug_path = '/mnt/cpath2/lf/data/fold2.txt'
        with open(debug_path, 'a+') as f:
            f.write('minipatch:' + str(mini_patch) + 'epoch:' + str(epoch))
            f.write('\n')
        start = time()
        print(f"wsi:[{wsi_name}], epoch:[{epoch}]:")
        input_img = wsi_img.to(args.device0)

        #分着来 
        buffer = []
        bs = input_img.size()[0]
        single_step = 300
        start = 0
        for i in range(int(bs / single_step) + 1):
            if start + single_step < bs:
                _, node_fea, _ = backbone(input_img[start:start+single_step], input_img[start:start+single_step],input_img[start:start+single_step], 0.75)
            else:
                _, node_fea, _ = backbone(input_img[start:], input_img[start:],input_img[start:], 0.75)
            start += single_step
            buffer.extend(node_fea)
        node_fea = torch.stack(buffer)
        save_path = os.path.join(save_folder, 'wsi43'+'.pth')
        torch.save(node_fea, save_path)


if __name__ == '__main__':
    
       



