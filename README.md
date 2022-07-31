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

# 实现细节
## utils.py
##  **chooseNodeMask**
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
```python

```

## **chooseEdgeMask**
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
    类内中心:这时候就要用到sort_idx_rst来判断统一类间两个点的距离了，使用如下函数来判断是否
```python
def judge_rad(u, v, sort_idx, rate):
    top = sort_idx[:int(len(sort_idx) * rate)]
    last = sort_idx[-int(len(sort_idx) * rate):]
    return True if ((u in top) and (v in last)) or((u in last) and (v in top)) else Fals

```
        2. 类内半径
        3. 类内随机
    