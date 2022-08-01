# 整体文件结构
*  ## maintrain
    * ### models
        所有的图神经模型都在这里
    * ### utils
        * utils.py

            里面包含有关聚类，mask点，mask边，计算特征中心和物理中心的函数
        * fold.py

            用于记录折叠的函数
    * ### construct_graph.py
        所有与建图有关的函数都在这里面

# utils.py
##  **chooseNodeMask**
### **usage**
    用于选择要添加mask的节点

### **args:**
    node_fea: 所有点的node_fea,包括折叠后的点
    cluster_num: 聚类的种类
    maskrate: list,用于存放三种相似度的mask比例。[高，中，低]
    wsi: 所有wsi的信息{idx:[pos, name, label]}
    stable_dic:自定义的数据结构，间fold.py，用于存储这次要更新的点
    cluster_res：前面聚类的结果。

### **returns:**
    mask_node_idx: 被mask的点的idx
    high: 与聚类中心相似度高的点
    low：与聚类中心相似度低的点
    sort_idx_rst：每一个cluster按照相似度排序的结果 [[id,id,....],[],[],[] ...]
    cluster_center_list: 每一类的聚类中心的特征
### 算法流程
    1. 首先将node_fea和node_idx按照聚类的结果分开，并提前计算好物理中心和边缘的点
```python
    node_fea_list, node_idx_list = split2clusters(node_fea, cluster_num, cluster_res, device, cluster_method)
    sort_idx_rst = [[] for i in range(cluster_num)]#用于存放每一类按照相似度从大到小的排序结果，后面edgemask的时候要用。
    cluster_center_list = []
    #取mask前先要判断是否重合
    pys_center, pys_edge = compute_pys_feature(wsi=wsi, n = 1)#计算处于物理中心和边缘的点
```
    2. 对每一类的点，先算出聚类中心，然后计算所有点和聚类中心的距离,并将点的index和相似度按照同样的顺序排序
```python
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
        sorted_disrt, sorted_idex = samesort(dist, idxs)#该函数实现了将数组2按照数组1同样的方式排序
        sort_idx_rst[idx].extend(sorted_idex)
```
    3.然后根据不同类别的mask rate对高相似度，中相似度和低相似度分别添加mask
```python
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
```

## **chooseEdgeMask**
### **usage**
    用于选择要添加mask的边
### **args:**
    u_v_pair:由源节点和目标节点组成的元组，形式如 ((id, id, ...), (id, id, ...))
    sort_idx_rst:每一类中的indx按照相似度从小到大的顺序排序，format:[[],[],....]
    rates:各种mask的比例共四类，类间，类内半径，类内中心，类内随机。用字典的形式传入。
### **returns:**
    edge_idx: 被mask的边的标号。
### 算法流程
    1. 首先按照源节点和目标节点是否为一类，将边分为两类，
```python
    diff = [] # 用于存储类间
    same = [[] for _ in range(len(set(clus_label)))] # 用于存储类内
```

    2. 对于类间，十分简单，先将所有的类间的边shuffle，然后取前n个就相当于随机了
```python
    random.shuffle(diff)
    mask_edge_pair.extend(diff[:int(len(u) * rates['inter'])])
```
    3. 对于类内，分为:类内中心，类内半径，类内随机三类
    3.1 类内半径:这时候就要用到sort_idx_rst来判断统一类间两个点的距离了，使用如下函数来判断
    主要思路就是在每一类按照相似度排序的列表中寻找前n%和后n%
```python
    def judge_rad(u, v, sort_idx, rate):
        top = sort_idx[:int(len(sort_idx) * rate)]
        last = sort_idx[-int(len(sort_idx) * rate):]
        return True if ((u in top) and (v in last)) or((u in last) and (v in top)) else Fals

```
    3.2 类内中心:思路和上面相似，就是找同一类且源和目标都在前n%的
```python
    def judge_center(u, v, sort_idx, rate):
        top = sort_idx[:int(len(sort_idx) * rate)]
        return True if (u in top) and (v in top) else False

```
    3.3 类内随机： 将类内的所有边shuffle，然后随机取
```python
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
```

## **neighber_type**
### **usage**
用于判断一个节点的周围n圈邻居的种类(聚类得出的种类)
### **arg:**
    pos (tuple(x, y)): 点的坐标
    n (int): 几圈邻居
    pos_dict: 用于存储所有点信息的字典{(x, y): label}
### **returns:**
    邻居的种类

### **算法流程**
    根据一个点的物理坐标，去遍历他的周围一圈邻居，看他周围一圈的邻居是否在wsi_dict当中，若在便查询其聚类的标签。

## **compute_pys_feature**
### **arg:**
    wsi (_type_): 建图时候使用的wsi结构体， {idx: (name, (x, y), ndoe_fea, (x_true, y_true), label)}
    n:查找周围n全邻居
### **returns:**
    center_nodes：位于中心的点,list of nodes的形式.
    edge_nodes: 位于边缘的点,list of nodes的形式.
### **算法流程**
    遍历所有的点，根据点周围n圈邻居的数量判断这个点的物理性质 (两层for循环遍历x, y轴)
```python
 for i in range(pos[0]-n, pos[0]+n+1):
        for j in range(pos[1]-n, pos[1]+n+1):
            if (i, j) in pos_dict:
                neighbers.append(pos_dict[(i,j)])
```



## **class minipatch**
### **usage**
    结构体，用于最后大patch合并用
### params:
    self.clus_center_list:用于存放这个大patch的聚类中心们
    self.clus_truelabel：用于存放每个大patch中所有点的真实标签

## **merge_mini_patch**

### **usage**
    用于合并大patch

### **arg:**
    patches_list：存储了所有大patch的聚类结果，每一个大patch的聚类结果为一个字典，key是聚类中心，val是这些点的真实种类，用于评价用[{cluster_center: [true_label, ...]}, {}, {}, {}, ...]
    thresholed：聚类中心的相似度的阈值

### **returns:**
    minipatch：最终经过合并的minipatch

### **算法流程**
    遍历要聚合的两个大patch，分别比较他们的聚类中心的相似度，若相似度够就将两个类的点合并，新聚类中心取平均值，若不够就不合并，单独将两个类再次添加到列表当中。

```python
    for i in range(a.clus_num):
            sim_buffer = []#用于存储a[i]和b的所有相似度，万一有多个阈值以上的，就比较并挑选值最大的那个
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
            #否则就分别加进去
            else:
                temp_center_list.append(a.clus_center_list[i])
                temp_label_list.append(a.clus_truelabel[i])
        #最后将b中和a没有相似度的也加进去
        temp_center_list.extend(b.clus_center_list)
        for i in b.clus_truelabel:
            temp_label_list.append(i)
```


# fold.py
## **class fold_dict**
### **usage**
    用于记录在过程中所有被折叠的点的index。
### **methods:**
    add_element():
        负责向字典中加与被折叠的点添加之前要判断一下有无已经是折叠后的新点，若是的话就要在新点下面添加,同时将旧的点放到新的里面，并置空，若不是则创建新点,在创建图的时候只要判断新加的点是否为空就行了，为空的话就不用添加边了。
    