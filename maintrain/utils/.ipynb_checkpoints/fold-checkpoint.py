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
class fold_dict:
    def __init__(self, num_nodes) -> None:
        """fold_dict:用于存储每个折叠后新产生的点是由哪些点组成
            next_id: 下一个新产生的被折叠的点的id
        """
        self.fold_dict = {}
        self.next_id = num_nodes
    def add_element(self, new_node:list):
        """负责向字典中加与被折叠的点
            添加之前要判断一下有无已经是折叠后的新点，若是的话就要在新点下面添加,同时将旧的点放到新的里面，并置空
            ，若不是则创建新点,在创建图的时候只要判断新加的点是否为空就行了，为空的话就不用添加边了。
        Args:
            node (list): 被折叠的点的list
        """
        nn = new_node
        #先判断是新产生一个折叠的点，还是像已有的点中添加信息
        re_fold_nodes = set(new_node) & set(self.fold_dict.keys())
        if len(re_fold_nodes) == 0: #新产生
            self.fold_dict[self.next_id] = nn
            self.next_id +=1
        else: #将旧的点添加进去
            #先将旧的点加进去
            self.fold_dict[self.next_id] = []
            for i in re_fold_nodes:
                self.fold_dict[self.next_id].extend(self.fold_dict[i])
                self.fold_dict[i] = []
                for idx, j in enumerate(nn):
                    if j == i:#去除该元素
                        nn.pop(idx)
            self.fold_dict[self.next_id].extend(nn)
            self.next_id +=1
class stable_dict:
    """format
        {clus_label:[nodes]}
    """
    def __init__(self, threshold) -> None:
        self.stable_dic = {}
        self.threshold = threshold
    def add_stable_idx(self, fes_center, pys_center, clus_label):
        """维护一个stable_list，用于按照累别存放不同类中稳定的点，这些点将在最后被折叠
            如果连续超过n次(要包含最后一次)都出现的话就算稳定
            v2.0 为了简单起见，目前先每mask一下就折叠一次

        Args:
            fes_center (_type_): _description_
            pys_center (_type_): _description_
        """
        
        stable_idx = list(set(fes_center) & set(pys_center))
        self.stable_dic[clus_label] = stable_idx
        # for idx in stable_idx:
        #     if idx in self.stable_dic.keys():
        #         self.stable_dic[idx] += 1
        #     else:
        #         self.stable_dic[idx] = 1
    def reset(self):
        self.stable_dic = {}
    def get_stable_nodes(self):
        returnlist = []
        for i in self.stable_dic.keys():
            if self.stable_dic[i] >= self.threshold:
                returnlist.append(i)
        return returnlist


def update_fold_dic(stable_dic: stable_dict, fold_dic: fold_dict):
    """根据stble_dic来更新fold_dic
        更新完成后stable_dic清零
    """
    for sta in stable_dic.stable_dic.keys():
        fold_dic.add_element(stable_dic.stable_dic[sta])
    stable_dic.reset()
    