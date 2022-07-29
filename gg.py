import torch 
import os
import numpy as np
from collections import Counter
saving_path = '/root/autodl-tmp/training_wsi'
dict_load_path = os.path.join(saving_path, ('43' + '.npy'))
wsi_dict =  dict(np.load(dict_load_path, allow_pickle='TRUE').item())
label = []
print(wsi_dict[0])
for i in range(len(wsi_dict)):
    label.append(wsi_dict[i][3])
print(Counter(label))