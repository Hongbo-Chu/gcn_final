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

# 1. utils.py
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
***

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
***
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

***
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


***
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


# 2. fold.py
## **class fold_dict**
### **usage**
    用于记录在过程中所有被折叠的点的index。
### **methods:**
* **compute_fold_id**

    根据要折叠的所有点的node_fea计算这些点折叠后应该落在哪个实体的点上面
    1. 先计算出所有点的聚类中心(不一定是一个实际的点)
    ```python
        node_fea_toupdate = []
        for idx in nodes_id:
            node_fea_toupdate.append(node_fea[idx])
        node_fea_toupdate = torch.stack(node_fea_toupdate)
        clus_center = node_fea_toupdate.mean(dim=0)
    ```
    2. 然后从实际的点当中找出与这个点相似度最高的实体点
    ```python
        cc = torch.stack([clus_center for _ in range(len(nodes_id))])
        dis = F.pairwise_distance(cc.unsqueeze(0), node_fea_toupdate.unsqueeze(0), p=2)
        dis_list = list(dis)
        max_idx = dis_list.index(max(dis_list))
        center_node = nodes_id[max_idx]
    ```
    3. 最后返回实际点的物理坐标，和聚类中心的node_fea,并在后面用这个新算出来的node_fea更新那个近似的实体的中心点
    ```python
        return center_node, clus_center
    ```
***

* **add_element():**
    
    用于向fold_dict当中添加元素

    新版本:折叠后的新点不再是新产生的，而是已有的点中和他相似度最高的点。

    1. 上来先判断要添加的点中有没有已经是被折叠的点(通过python集合的方式)
    ```python
        re_fold_nodes = set(new_node) & set(self.fold_dict.keys()) 
    ``` 
    2. 如果没有，就不涉及到合并，直接将所有的点添加到折叠字典当中去。
    ```python
        if len(re_fold_nodes) == 0: #新产生
        #计算折叠后的点的新id，从所有点中选出一个相似度最大的点
        center_node, center_node_fea = self.compute_fold_id(nn, node_fea) # 计算出最近似的实体的点，以及新的折叠中心的node_fea
        self.fold_dict[center_node] = nn
        self.fold_node_fea[center_node] = center_node_fea
    ```
    3. 如果有，就要把旧的折叠的所有的点清空，然后添加到新的折叠点的集合当中，并重新计算折叠中心。
    ```python
        else: #若果新折叠的点中包含已经被折叠的点
        node_tobe_fold = []
        for i in re_fold_nodes: # 先将折叠字典中旧的点全部添加到临时变量中去，用于后续重新计算新的折叠中心
            node_tobe_fold.extend(self.fold_dict[i])
            node_tobe_fold.extend(i)
            ##同时旧的折叠字典中去除这些点
            self.fold_dict.pop(i)
        center_node, center_node_fea = self.compute_fold_id(node_tobe_fold, node_fea)
        self.fold_dict[center_node] = node_tobe_fold
        self.fold_node_fea[center_node] = center_node_fea
    ```
***
## **update_fold_dic**
### **usage**
    根据stable_dict来更新fold_dict,更新完成后stable_dict清零
### **args**
    stable_dic: 每轮更新后稳定的点
    fold_dic: 折叠字典
    node_fea：
    参数的更新不通过返回，而是通过直接传参
### **returns**
    None: 通过直接传参

### **实现细节**
遍历所有的稳定字典，将每一个稳定字典加入折叠字典中
```python
     for sta in stable_dic.stable_dic.keys():
        if len(stable_dic.stable_dic[sta]) >= 2:
            fold_dic.add_element(stable_dic.stable_dic[sta], node_fea)
```

# 3. construct_graph.py

## **class new_graph**
### **usage:**
    根据物理信息和特征信息来创建图。
### **属性**
* self.edge_mlp: 用于将javed的多维的edge_fea映射到一维上去，方便后续阈值，拼接操作
* self.wsi_di：用于记录输入的wsi的基本信息，包括物理坐标，种类等等。
* self.fold_dict：一直维护的折叠字典
* self.d：javed公式中的d
* self.node_num：所有点的数量
* self.node_fea：点的特征
* self.edge_enhance：增强邻居节点的值

### **methods**
1.  **init_edge**

    usage: 用于初始化边的权重，参考javed方法

    returns: 任意两点之间的边的权重的矩阵
    
    实现细节：
    1. 首先参考javed公式，算出f_ij, 和d_ij。
    ```PYTHON
        h = w = 128
        f_ij = L2_dist(self.node_fea, self.node_fea)#公式中||f_i - f_j||_2
        d_ij = L2_dist(self.d, self.d)#公式中d_ij
        px =  self.d.permute(1, 0)[0] # 所有x坐标
        px1 = px.expand(px.size(0), px.size(0))
        px2 = px1.permute(1, 0)
        py = self.d.permute(1, 0 )[1] # 所有的y坐标
        py1 = py.expand(py.size(0), py.size(0))
        py2 = py1.permute(1, 0) # 
        p_ij1 = ((px1 - px2) / h)
        p_ij2 = ((py1 - py2) / h)
        #这里需要每个分量都是N * N
        self.node_num = len(self.node_fea)
        z = torch.zeros_like(f_ij).to(self.device)
        # print(f"各种大小{f_ij.size()} {p_ij1.size()} {p_ij2.size()} {d_ij.size()} {z.size()}")
        edge_fea = torch.stack([f_ij, p_ij1, p_ij2, z, z, d_ij])
    ```
    2. 使用permute操作将edge_fea转为 [n, n, edge_fea]，这样edge_fea[i][j]就代表了这两点之间的边的权重
    ```python
        edge_fea = edge_fea.permute(2, 1, 0)
    ```
    3. 由于建的是全连接图，通过for循环的形式将[n, n, edge_fea]转化为[n*n, edge_fea]
    ```python 
        for i in tqdm(range(self.node_num)): #全连接图
        for j in range(self.node_num):
            e_ij.append(edge_fea[i][j])
        e_ij = torch.stack(e_ij)
        return e_ij
    ```
2. **init_graph**

    usage: 用于初始化整个图。

    returns:
    
    * dgl.graph:下面计算用的图
    * (u, v): 源节点和目标节点对
    * ee：边的特征

    算法细节：

    1. 首先调用 `init_edge`方法生成边的权重，并使用 `edge_mlp`将其映射为一维，方便后续的计算
    ```python
        e_fea = self.init_edge()
        e_fea = self.edge_mlp(e_fea).view(self.node_num, self.node_num)#[n^2, 6] -> [n^2, 1] -> [n, n]
    ```
    2. 然后为了减少边的连接数量，使用阈值来卡掉一部分关系不那么近的边。
        阈值的设定就以所有的edge_fea为基准上下浮动。
    ```python
        threshold = e_fea.mea()
        threshold_e = torch.threshold(e_fea, threshold, 0)#size() = n,n
    ```
    3. 接下来对所有的邻居边进行增强
    ```python
        for node in range(len(self.wsi_dic)):
            temp = torch.zeros(1, self.node_num)
            neighbor_edge = neighbor_idx(node, self.wsi_dic, 1)
            #折叠的点不加强
            neighbor_edge.extend([0 for _ in range(len(list(self.fold_dict.keys())))])
            temp[0][neighbor_edge] = self.edge_enhance
            edge_enhance.append(temp)
    ```
    4. 下面根据算出来的edge_fea进行建图。建图建一半就行，另一半调用dgl接口对称生成
        1. 先统计一下所有被折叠的点，以便在后面把它去掉
        ```PYTHON
            fold_nodes = []
            for i in self.fold_dict.keys():
                fold_nodes.extend(self.fold_dict[i])
        ```
        2. 然后遍历建图。在建图的时候判断阈值不等于0，同时要记录所有已经被折叠点出现的位置，在后面去除这些点
        ```python
            for i in tqdm(range(self.node_num)): #全连接图
                for j in range(i+1 ,self.node_num):
                    if threshold_e[i][j] != 0 and i != j:#判断在阈值之内可以，且无自环
                        if i in fold_nodes or j in fold_nodes:#记录被折叠点的坐标，因为后面添加的点的连接要根据它都包含了哪些点决定
                            count_list.append(count)#不用记录具体信息，因为反正这些点都要去掉
                        u.append(i)
                        v.append(j)
                        ee.append((threshold_e[i][j]).unsqueeze(0))
                        count += 1
        ```
        3. 去除所有已经被折叠的点。这里按照idx倒叙去除，这样在去除后面的同时前面的idx不会被改变
        ```python
            count_list.reverse()
            for dele in count_list:
                ee.pop(dele)
                u.pop(dele)
                v.pop(dele)
            temp_graph = dgl.graph((u, v))
            self.graph = dgl.add_reverse_edges(temp_graph).to(self.device)
        ```
        4. 最后建双向图
        ```python
            temp_graph = dgl.graph((u, v))
            self.graph = dgl.add_reverse_edges(temp_graph).to(self.device)
            ee = torch.cat(ee, dim=0).unsqueeze(1)#最终的edge_fea，是将那些为0的边都去掉了
            ee = torch.cat([ee,ee], dim =0)
            return self.graph, (u, v), ee
        ```

# 4. loss.py
## **class my_loss**

### inner_cluster_loss
Args:

* node_fea (tensor): 更新后的node_fea，require_grade=True
* clu_label (_type_): 每个点的聚类标签
* center_fea (_type_): 几个聚类中心的向量,list of tensor的形式
* mask_nodes:加了mask的点
* mask_weight:对于mask点的聚类权重

实现细节：
1. 先建立空的列表用于存储每一类的每点和聚类中心的距离、
```python
    L2_dist = [torch.tensor([]).to("cuda:1") for _ in range(len(center_fea))]
    final_loss = torch.tensor(0).to("cuda:1")
    # print(f"loss检测{node_fea.size()}, {len(center_fea)}, {len(clu_label)}")
    center_fea = torch.stack(center_fea).to("cuda:1")
```
2. 根据聚类的结果计算每一类中的每一个点到该类聚类中心的距离
```python
    for i in range(len(node_fea)):
                L2_dist[clu_label[i]] = torch.cat([L2_dist[clu_label[i]], (F.pairwise_distance(node_fea[i].unsqueeze(0), center_fea[clu_label[i]].unsqueeze(0), p=2))])
                if  i in mask_nodes:
                    L2_dist[clu_label[i]] = torch.cat([L2_dist[clu_label[i]],  (1 + mask_weight) * F.pairwise_distance(node_fea[i].unsqueeze(0), center_fea[clu_label[i]].unsqueeze(0), p=2)])
```
3. 对每一类的距离进行归一化操作
```python
    for i in range(len(L2_dist)):
        if len(L2_dist[i]) != 0:
            L2_dis_min = L2_dist[i].min()   
            L2_dist[i] = L2_dist[i] - L2_dis_min
            L2_dist_max = L2_dist[i].max()
            L2_dist[i] = L2_dist[i] / L2_dist_max
            L2_dist[i] = L2_dist[i].mean()
            final_loss = final_loss + L2_dist[i]
    return final_loss
```
***
### inter_cluster_loss
Args:
* node_fea (_type_): 节点特征
* clu_label (_type_): 不需要了现在，都包含在sort_idx_rst中了。
* center_fea (_type_): 每一类的聚类中心
* sort_idx_rst (_type_):每类的相似度排序结果[[],[],[]....]
* mask_nodes (_type_): 加了mask的点
* mask_weight (_type_): 对于mask点的聚类权重
* center_fea: 每一类的聚类中心list of tensor, [dim]

实现细节：
思路就是两层for循环遍历所有的类，两两算类间loss
1. 先将每一类中的所有点的`node_fea`找出来，用于后面计算距离用
```python
    edgenode_i_idx = sort_idx_rst[i][-int((len(sort_idx_rst[i])) * 0.1):]
    edgenode_j_idx = sort_idx_rst[j][-int((len(sort_idx_rst[j])) * 0.1):]
    edgenode_fea_i = [node_fea[i] for i in edgenode_i_idx]
    edgenode_fea_j = [node_fea[j] for j in edgenode_j_idx]
```
2. 然后计算这些点与两个高维球两线的pairwise_cos角度
```python
    t = (center_fea[i] - center_fea[j]).unsqueeze(0)
    cos_i = [F.cosine_similarity(t, k.unsqueeze(0)) for k in edgenode_fea_i]
    cos_j = [F.cosine_similarity(t, w.unsqueeze(0)) for w in edgenode_fea_j]
```
3. 找出两个高维球之间符合条件的点。
    * 为了防止没有点的cos角度在输入的阈值之内，这里采用了自适应的阈值选择方法。`th_i`和`th_j`就是选择的阈值,是根据所算出来的所有的cos角进行排序。
    * 然后将所有的符合条件的点的`ndoe_fea`挑出来
```python
    sorted_cos_i = sorted(cos_i)
    th_i = sorted_cos_i[int(len(sorted_cos_i) * 0.05)]
    sorted_cos_j = sorted(cos_j)
    th_j = sorted_cos_j[int(len(sorted_cos_j) * 0.05)]
    for k, cosval in enumerate(cos_i):
        if cosval > th_i:
            final_i.append(edgenode_fea_i[k].unsqueeze(0))
    for q, cosval in enumerate(cos_j):
        if cosval > th_j:
            final_j.append(edgenode_fea_j[q].unsqueeze(0))

```
4. 计算并累计所有的类间距离
```pyhton
    L2_dist += F.pairwise_distance(final_i.unsqueeze(0), final_j.unsqueeze(0), p=2)
```

# 5. trainengine_modelparallel.py
## freeze & unfreeze
### **usage:**
用于冻结backbone的梯度，使其不进行训练
### **args:**
* 三个model
* optimizer
### **returns**
更新后的optimizer。
### 实现细节：
* 将`bakckbone`的所有参数的`require_grade`设置为`False`  
* 将optimizer中backbonde的参数去掉
***

## train_one_wsi

### **usage:**
一个wsi的一个大patch的一组训练
### **实现细节**
1. 首先初始化折叠字典，并将模型设置为训练模式
```python
    backbone.train()
    gcn.train()
    wsi_name = wsi_dict[0][0].split("_")[0]
    fold_dic = fd()#用于记录被折叠的点
    stable_dic = sd()#用于记录稳定的点
```
2. 根据超参数中的每个大patch的训练数量设置循环
```python
for epoch in range(args.epoch_per_wsi):
```
3. 先根据当前轮数判断是否需要冻结backbone。
```python
if (epoch+1) % 3 == 0:
    optimizer = freeze(backbone, gcn, graph_mlp, args)
else:
    optimizer = unfreeze(backbone, gcn, graph_mlp, args)
```
4. 计算node_fea，随后创建一个从计算图中剥离的副本用于进行edge_fea的计算。并跟更新折叠字典，并根据折叠字典将折叠后的node_fae进行更新。(第一轮字典为空，不进行更新)
```python
    node_fea = backbone(input_img)
    node_fea_detach = node_fea.clone().detach()#从计算图中剥离
    update_fold_dic(stable_dic, fold_dic, node_fea_detach)
    for k in fold_dic.fold_dict.keys():
        node_fea[k] = fold_dic.fold_node_fea[k]
```
5. 建图，算聚类的结果，用于选择mask,并将算出来的聚类结果添加到wsi_dict中，方便后边函数使用。
```python
    g, u_v_pair, edge_fea = new_graph(wsi_dict, fold_dic, node_fea_detach, args.edge_enhance, graph_mlp, args.device1).init_graph()     
    clu_label, clus_num = Cluster(node_fea=node_fea_detach, device=args.device1, method=args.cluster_method).predict(threshold_dis=args.heirachi_clus_thres)
    for i in range(len(wsi_dict)):
        wsi_dict[i].append(clu_label[i])
```
6. 根据`mask_rate`计算`node_mask`和`edge_mask`
```python
    mask_rates = [args.mask_rate_high, args.mask_rate_mid, args.mask_rate_low]#各个被mask的比例
    mask_idx, fea_center, fea_edge, sort_idx_rst, cluster_center_fea = chooseNodeMask(node_fea_detach, clus_num, mask_rates, wsi_dict, args.device1, stable_dic, clu_label)#TODO 检查数量
    mask_edge_idx = chooseEdgeMask(u_v_pair, clu_label,sort_idx_rst, {"inter":args.edge_mask_inter, "inner":args.edge_mask_inner, "random": args.edge_mask_random} )#类内半径多一点
    node_fea[mask_idx] = 0
    edge_fea[mask_edge_idx] = 0
```
7. 模型并行，将`graph`和`node_fea`, `edge_fea`放到`cuda:1`上面。然后过`gcn`,并计算`loss`，进行梯度更新。
```python
    g = g.to(args.device1)
    edge_fea = edge_fea.to(args.device1)
    node_fea = node_fea.to(args.device1)
    predict_nodes = gcn(g, node_fea, edge_fea)
    loss = criterion(predict_nodes, clu_label, cluster_center_fea, mask_idx, args.mask_weight, sort_idx_rst)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(
```
8. 使用更新后的特征重新聚类。
```python
    pys_center, pys_edge = compute_pys_feature(wsi_dict, args.pos_choose) #计算物理特征
    # fea2pos(fea_center, fea_edge, pys_center, pys_edge)#统计对齐信息并打印
    predict_nodes_detach = predict_nodes.clone().detach()
    clu_labe_new, _ = Cluster(node_fea=predict_nodes_detach, device=args.device1, method=args.cluster_method).predict(threshold_dis=args.heirachi_clus_thres)
    for i in range(len(wsi_dict)):
```


    
        

