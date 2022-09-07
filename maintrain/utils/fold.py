"""
用于折叠操作
整个循环中的折叠操作分为两部分：对折叠的判断，和进行折叠。
同时整个过程中，维护稳定表，和折叠树两个数据结构
1. 折叠的判断：
    要折叠的点是在更新过程中几轮都重合(稳定)的点，因为需要一张稳定表来记录一个点稳定的次数
    若稳定的次数满足条件，则在n次循环之后进行折叠。
2. 进行折叠：
    折叠的操作分为更新node_fea矩阵，和维护折叠字典
    a. 更新node_fea矩阵:
        根据更新表将折叠完的点直接添加这原有的tensor后面，旧点的特征不做删除，因为要维护节点的idx不变。
        只需要在建图的时候对要更新的点做出忽略，使其成为孤点即可，这样在消息传递中就不会更新，跟删除没有区别了就。
    b. 维护折叠字典：#TODO 
        每一类对应一颗折叠字典

"""
from enum import Flag
from pickle import FALSE
import torch
import torch.nn.functional as F
import copy
class fold_dict:
    """
    最新版本，将fold_dic和stable_dic合并！
    方便统计已经折叠的字典，使得不再统计已经折叠完的点
    """
    def __init__(self, n) -> None:
        """fold_dict:用于存储每个折叠后新产生的点是由哪些点组成
            next_id: 下一个新产生的被折叠的点的id
        """
        self.fold_dict = {}
        self.fold_node_fea = {}
        self.fold_list = [] # fold_dict的列表形式，将所有的点放在一个列表里面，方便查找
        self.num_stable = n # 稳定n轮再加进去
        self.cluster_num_lastepoch = 0 # 上一个epoch的聚类个数 初始化为零
        self.display = []
        self.fold_dic_is_empty = True # 用于记录被清空后第一次添加元素
        #---------------------------原stable_dic------------------------------------------------------
        self.stable_dic = {} # node_tobe_fold
        self.cluster_num_lastepoch = 0 # 上一个epoch的聚类个数 初始化为零
    def compute_fold_id(self,nodes_id, node_fea):
        """
            根据要折叠的所有点的node_fea计算这些点折叠后应该落在哪个实体的点上面

            todo只看特征维度
        """
        node_fea_toupdate = []  # repeat
        for idx in nodes_id:
            node_fea_toupdate.append(node_fea[idx])
        node_fea_toupdate = torch.stack(node_fea_toupdate)
        clus_center = node_fea_toupdate.mean(dim=0)
        #然后计算所有的点和这个算出来的中心的相似度
        # cc = torch.stack([clus_center for _ in range(len(nodes_id))])
        cc = clus_center.repeat(len(nodes_id), 1)
        dis = F.pairwise_distance(cc.unsqueeze(0), node_fea_toupdate.unsqueeze(0), p=2)
        dis_list = list(dis)
        max_idx = dis_list.index(max(dis_list))
        center_node = nodes_id[max_idx]
        return center_node, clus_center

    def sync_fold_dic_list(self):
        """
        用于同步fold_dic和fold_list中的内容
        """
        for k, v in self.fold_dict.items():
            self.fold_list.extend(v)
        self.fold_list = list(set(self.fold_list))

    def add_fold_element(self, new_node:list, node_fea):
        """负责向字典中加与被折叠的点
            添加之前要判断一下有无已经是折叠后的新点，若是的话就要在新点下面添加,同时将旧的点放到新的里面，并置空
            ，若不是则创建新点,在创建图的时候只要判断新加的点是否为空就行了，为空的话就不用添加边了。

            v2.0： 折叠后的新点不再是新产生的，而是已有的点中和他相似度最高的点。
            v2.1: 添加新的cluster的时候，是根据这个cluster和已有的fold_cluster中的相似程度来判断添加到哪个当中去。
            v2.2: 新点稳定n轮之后才加进去，加进去之在stabledic中就不会出现了。
        Args:
            node (list): 被折叠的点的list
        """
        nn = new_node
        #分两种情况
            #1. fold_dic是空的，往里面添加新的元素
            #2. fold_dic里面已经有折叠的cluster了，向cluster中添加元素
        if self.fold_dic_is_empty:
            #fold_dic是空的，直接添加元素
            center_node, center_node_fea = self.compute_fold_id(nn, node_fea)
            self.fold_dict[center_node] = nn
            self.fold_node_fea[center_node] = center_node_fea
            # print(f"真的更新了呢{self.fold_node_fea}")
            # assert False
        else:
            #向已经有的fold_dic中添加，通过特征维度的距离来判断添加到哪一类当中
            #分别计算每个点和特征中心的距离
            #添加完成后再重新计算特征中心，看是否更新
            #先统计一下中心点的特征向量
            clus_center_fea = []
            clus_center_fea_key = []
            for k, v in self.fold_node_fea.items():
                # 保证k,v一一对应
                clus_center_fea_key.append(k)
                clus_center_fea.append(v)
            try:
                clus_center_fea = torch.cat(clus_center_fea, dim=0)
            except:
                print(clus_center_fea)
                print(self.fold_node_fea)
                assert False, "服了"
            for new_fold_node in nn:
                try:
                    dis_to_centers = F.pairwise_distance(torch.tensor(new_fold_node).repeat(len(self.fold_node_fea), 1).to("cuda:0"), clus_center_fea.unsqueeze(0), p=2)

                except:
                    print((torch.tensor(new_fold_node).repeat(len(self.fold_node_fea), 1)).size())
                    print(torch.tensor(clus_center_fea).unsqueeze(0).size())
                    assert False, 'eee'
                #找出最近的距离的index
                min_dis_idx = torch.argmin(dis_to_centers)
                fold_cluser_to_append = clus_center_fea_key[min_dis_idx]
                #将新的点添加到这个cluster里面
                try:
                    self.fold_dict[fold_cluser_to_append].append(new_fold_node)
                    #重新计算一下折叠的中心

                except:
                    print(self.fold_dict)
                    print(fold_cluser_to_append)
                    print(new_fold_node)
                    print("*"*100)
                    print(self.fold_dict[fold_cluser_to_append])
                    print(type(self.fold_dict[fold_cluser_to_append]))
                    assert False, 'qqq'
        #最后同步一下fold_dic和fold_list
        self.sync_fold_dic_list()

    def update_fold_dic(self, node_fea, clus_num):
        """根据stble_dic来更新fold_dic
            更新完成后stable_dic清零
        """
        self.display.append(clus_num)
        print(f"类别数：{self.display}")
        # print("更新折叠字典")
        old_dic = copy.deepcopy(self.fold_dict)
        # print(stable_dic.stable_dic)
        # print("*"*100)
        # print(fold_dic.fold_dict)
        fold_path = '/root/autodl-tmp/debuging/fold2.txt'
        with open(fold_path, 'a+') as f:
            f.write('stable_dic')
            f.write(str(self.stable_dic))
            f.write('\n')
            f.write('folddic:')
            f.write(str(self.fold_dict))
            f.write('\n')
            f.write('\n')
        #更新内容：
        # 保存不变
        print(clus_num)
        print(self.cluster_num_lastepoch)
        if clus_num == self.cluster_num_lastepoch:
            flag = False # 有可能并没有可更新的，只有更新了fold_dic_is_empty才变
            for sta in self.stable_dic.keys():
                #先选出稳定n轮的点
                stable_nodes = []
                stable_idx = [idx for (num, idx) in zip(self.stable_dic[sta][1], self.stable_dic[sta][0]) if num == self.num_stable]
                #加到fold_dic中
                if len(stable_idx) != 0:
                    flag = True
                    self.add_fold_element(stable_idx, node_fea)
            if flag:
                self.fold_dic_is_empty = False

        #聚类种类变了
        else:
            # 在聚类种类发生变化的时候，stable_dic清空，fold在这轮肯定无法更新(因为需要等稳定几轮)，同时也一并清空
            self.reset_fold_dic()
        print(f"folddic:{self.fold_dict}")


        # for sta in stable_dic.stable_dic.keys():
        #     #先选出稳定n轮的点
        #     stable_nodes = []
        #     stable_idx = [idx for (num, idx) in zip(sta[1], sta[0]) if num == self.fold_dic.num_stable]
        #     #加到fold_dic中
        #     self.fold_dic.add_element(stable_idx, node_fea)
        #最后再更新聚类数量的记录
        self.cluster_num_lastepoch = clus_num
        new_dic = {}
        for k_new, v_new in self.fold_dict.items():
            if k_new in list(old_dic.keys()):
                new_dic[k_new] = list(set(self.fold_dict[k_new]) - set(old_dic[k_new]))
            else:
                new_dic[k_new] = self.fold_dict[k_new]
        with open(fold_path, "a+") as f:
            for k, v in new_dic.items():
                if v != []:
                    # print(f"中心点{k}, 更新了{v}")
                    f.write("中心点"+str(k) + "更新了" + str(v))
                    f.write('\n')
            f.write("*"*100)
            f.write('\n')
            f.write('\n')
            f.write('\n')
        # stable_dic.reset_stable_dic()
    def reset_fold_dic(self):
        self.fold_dict = {}
        self.fold_node_fea = {}
        self.fold_dic_is_empty =True
    def ensure_clus_num(self, cluster_num):
        """用于判断每轮的聚类数量是否发生了变化

            如果不变，就接着之前的折叠字典更新
            要是变了，就清空原来的折叠字典，重新更新
            好处就是，如果聚类的数量一直在变的话(不稳定)，就不进行点的折叠。
        """
        if self.cluster_num_lastepoch == cluster_num:
            pass
        else:
            print("stable_dic has been reset")
            self.reset_stable_dic()
        self.cluster_num_lastepoch = cluster_num
    def add_stable_idx(self, fes_center, pys_center, clus_label):
        """维护一个stable_list，用于按照累别存放不同类中稳定的点，这些点将在最后被折叠
            如果连续超过n次(要包含最后一次)都出现的话就算稳定
            v2.1 添加对于稳定次数的记录
            新数据结构：{clus_idx:[[idx, idx, idx,...], [num, num, num,...]]}
            用于记录可能被折叠的点的稳定的次数
                1. 旧的没有出现的点要被踢出去，
                2. 重复出现的点加一
                3. 新来的点置一
            同时，已经在fold_dic当中的点就不会再添加到stable_dic里面
            由于stable_dic每次的计算都是不受控制的，所以需要已经在fold_dic中的的点来判断
            
        Args:
            fes_center (_type_): _description_
            pys_center (_type_): _description_
        """
        
        new_idx = list(set(fes_center) & set(pys_center))
        # print(set(fes_center))
        # print(set(pys_center))
        # print(f"要添加这些点{stable_idx}")
        #找出stable_dict中的同一类
        #同时也有可能产生一个新的聚类中心
        count = 0
        key = -1
        for k, v in self.stable_dic.items():
            intersection = set(v[0]) & set(new_idx) #计算交集
            # print(f"烫烫烫烫烫烫烫烫烫{intersection} ,{v[0]}")
            # print(f"bbbbb{new_idx}")
            if len(list(intersection)) > count:
                count = len(list(intersection))
                key = k
        if key != -1:
            """不是新产生的cluster
                将intersection 和 new_node 拼接一下
                要先吧intersection挑出来
            """
            new_stable_dict_idx = [] # 新的稳定字典
            new_stable_dict_num = [] # 稳定的次数
            #要刨去已经被折叠的点 - set(self.fold_list)
            new_stable_dict_idx.extend(list((set(self.stable_dic[key][0]) & set(new_idx)) - set(self.fold_list)))
            new_nodes = (set(new_idx) - set(self.stable_dic[key][0])) - set(self.fold_list)
            stable_num = []
            for i in new_stable_dict_idx:
                idx = self.stable_dic[key][0].index(i)
                new_stable_dict_num.append(self.stable_dic[key][1][idx])
            #stable_num + 1 
            new_stable_dict_num = [i+1 for i in new_stable_dict_num]
            #将这轮新的点加进来
            new_stable_dict_idx.extend(list(new_nodes))
            new_stable_dict_num.extend([1 for _ in range(len(new_nodes))])
            self.stable_dic[key] = [new_stable_dict_idx, new_stable_dict_num]

        else:
            """是新的cluster
            
            """
            new_key = len(list(self.stable_dic.keys())) + 1
            self.stable_dic[new_key] = [new_idx, [1 for _ in range(len(new_idx))]]
        # for idx in stable_idx:
        #     if idx in self.stable_dic.keys():
        #         self.stable_dic[idx] += 1
        #     else:
        #         self.stable_dic[idx] = 1
    def reset_stable_dic(self):
        self.stable_dic = {}
    def get_stable_nodes(self):
        returnlist = []
        for i in self.stable_dic.keys():
            if self.stable_dic[i] >= self.threshold:
                returnlist.append(i)
        return returnlist

# class stable_dict(fold_dict):
#     """format
#         {clus_label:[nodes]}
#     """
#     def __init__(self) -> None:
#         self.stable_dic = {}
#         self.cluster_num_lastepoch = 0 # 上一个epoch的聚类个数 初始化为零
#     def ensure_clus_num(self, cluster_num):
#         """用于判断每轮的聚类数量是否发生了变化

#             如果不变，就接着之前的折叠字典更新
#             要是变了，就清空原来的折叠字典，重新更新
#             好处就是，如果聚类的数量一直在变的话(不稳定)，就不进行点的折叠。
#         """
#         if self.cluster_num_lastepoch == cluster_num:
#             pass
#         else:
#             print("stable_dic has been reset")
#             self.reset_stable_dic()
#         self.cluster_num_lastepoch = cluster_num
#     def add_stable_idx(self, fes_center, pys_center, clus_label):
#         """维护一个stable_list，用于按照累别存放不同类中稳定的点，这些点将在最后被折叠
#             如果连续超过n次(要包含最后一次)都出现的话就算稳定
#             v2.1 添加对于稳定次数的记录
#             新数据结构：{clus_idx:[[idx, idx, idx,...], [num, num, num,...]]}
#             用于记录可能被折叠的点的稳定的次数
#                 1. 旧的没有出现的点要被踢出去，
#                 2. 重复出现的点加一
#                 3. 新来的点置一
#             同时，已经在fold_dic当中的点就不会再添加到stable_dic里面
#             由于stable_dic每次的计算都是不受控制的，所以需要已经在fold_dic中的的点来判断
            
#         Args:
#             fes_center (_type_): _description_
#             pys_center (_type_): _description_
#         """
        
#         new_idx = list(set(fes_center) & set(pys_center))
#         # print(set(fes_center))
#         # print(set(pys_center))
#         # print(f"要添加这些点{stable_idx}")
#         #找出stable_dict中的同一类
#         #同时也有可能产生一个新的聚类中心
#         count = 0
#         key = -1
#         print(f"cpcpcp{self.stable_dic}")
#         for k, v in self.stable_dic.items():
#             intersection = set(v[0]) & set(new_idx) #计算交集
#             # print(f"烫烫烫烫烫烫烫烫烫{intersection} ,{v[0]}")
#             # print(f"bbbbb{new_idx}")
#             if len(list(intersection)) > count:
#                 count = len(list(intersection))
#                 key = k
#         if key != -1:
#             """不是新产生的cluster
#                 将intersection 和 new_node 拼接一下
#                 要先吧intersection挑出来
#             """
#             new_stable_dict_idx = [] # 新的稳定字典
#             new_stable_dict_num = [] # 稳定的次数
#             new_stable_dict_idx.extend(list(set(self.stable_dic[key][0]) & set(new_idx)))
#             new_nodes = set(new_idx) - set(self.stable_dic[key][0])
#             stable_num = []
#             for i in new_stable_dict_idx:
#                 idx = self.stable_dic[key][0].index(i)
#                 new_stable_dict_num.append(self.stable_dic[key][1][idx])
#             #stable_num + 1 
#             new_stable_dict_num = [i+1 for i in new_stable_dict_num]
#             #将这轮新的点加进来
#             new_stable_dict_idx.extend(list(new_nodes))
#             new_stable_dict_num.extend([1 for _ in range(len(new_nodes))])
#             self.stable_dic[key] = [new_stable_dict_idx, new_stable_dict_num]

#         else:
#             """是新的cluster
            
#             """
#             new_key = len(list(self.stable_dic.keys())) + 1
#             self.stable_dic[new_key] = [new_idx, [1 for _ in range(len(new_idx))]]
#         # for idx in stable_idx:
#         #     if idx in self.stable_dic.keys():
#         #         self.stable_dic[idx] += 1
#         #     else:
#         #         self.stable_dic[idx] = 1
#     def reset_stable_dic(self):
#         self.stable_dic = {}
#     def get_stable_nodes(self):
#         returnlist = []
#         for i in self.stable_dic.keys():
#             if self.stable_dic[i] >= self.threshold:
#                 returnlist.append(i)
#         return returnlist


# def update_fold_dic(stable_dic: stable_dict, fold_dic: fold_dict, node_fea, clus_num):
#     """根据stble_dic来更新fold_dic
#         更新完成后stable_dic清零
#     """
#     # print("更新折叠字典")
#     old_dic = copy.deepcopy(fold_dic.fold_dict)
#     # print(stable_dic.stable_dic)
#     # print("*"*100)
#     # print(fold_dic.fold_dict)
#     fold_path = '/root/autodl-tmp/7.26备份/fold2.txt'
#     with open(fold_path, 'a+') as f:
#         f.write('stable_dic')
#         f.write(str(stable_dic.stable_dic))
#         f.write('\n')
#         f.write('folddic:')
#         f.write(str(fold_dic.fold_dict))
#         f.write('\n')
#         f.write('\n')
#     #更新内容：
#     for sta in stable_dic.stable_dic.keys():
#         #先选出稳定n轮的点
#         stable_nodes = []
#         stable_idx = [idx for (num, idx) in zip(sta[1], sta[0]) if num == fold_dic.num_stable]
#         #加到fold_dic中
#         fold_dic.add_element(stable_idx, node_fea)
#     new_dic = {}
#     for k_new, v_new in fold_dic.fold_dict.items():
#         if k_new in list(old_dic.keys()):
#             new_dic[k_new] = list(set(fold_dic.fold_dict[k_new]) - set(old_dic[k_new]))
#         else:
#             new_dic[k_new] = fold_dic.fold_dict[k_new]
#     with open(fold_path, "a+") as f:
#         for k, v in new_dic.items():
#             if v != []:
#                 # print(f"中心点{k}, 更新了{v}")
#                 f.write("中心点"+str(k) + "更新了" + str(v))
#                 f.write('\n')
#         f.write("*"*100)
#         f.write('\n')
#         f.write('\n')
#         f.write('\n')
#     stable_dic.reset()