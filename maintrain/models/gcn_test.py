from platform import node
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import dgl
import dgl.function as fn
from dgl.utils import expand_as_pair


# def L2_dist(x, y):
#     """
#     a fast approach to compute X, Y euclidean distace.
#     Args:
#         x: pytorch Variable, with shape [m, d]
#         y: pytorch Variable, with shape [n, d]
#     Returns:
#         dist: pytorch Variable, with shape [m, n]
#     """
#     #y: pytorch Variable, with shape [n, d]
#     m, n = x.size(0), y.size(0)
#     # xx经过pow()方法对每单个数据进行二次方操作后，在axis=1 方向（横向，就是第一列向最后一列的方向）加和，此时xx的shape为(m, 1)，经过expand()方法，扩展n-1次，此时xx的shape为(m, n)
#     xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
#     # yy会在最后进行转置的操作
#     yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
#     dist = xx + yy
#     # torch.addmm(beta=1, input, alpha=1, mat1, mat2, out=None)，这行表示的意思是dist - 2 * x * yT 
#     dist.addmm_(x, y.t(), beta=1, alpha=-2)
#     # clamp()函数可以限定dist内元素的最大最小范围，dist最后开方，得到样本之间的距离矩阵
#     dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
#     # 归一化
#     dist = torch.diag(dist)
#     # bn = torch.nn.BatchNorm1d(1, affine=False).to('cuda:1')
#     dis_min = dist.min()
#     dis_max = dist.max()
#     dist = (dist - dis_min) / dis_max
#     return dist

def L2_dist(x, y):
    # print(f"球球了{x.size()}")
    dis = F.pairwise_distance(x, y, p=2)
    dis_min = dis.min()
    dis = dis - dis_min
    dis_max = dis.max()
    dis = dis / dis_max
    return dis


# class graph_mlp(torch.nn.Module):
#     def __init__(self, in_dim, hid_dim, out_dim):
#         super(graph_mlp, self).__init__()
#         self.linear1 = torch.nn.Linear(in_dim, hid_dim)
#         self.linear2 = torch.nn.Linear(hid_dim, out_dim)
#         self.relu = torch.nn.ReLU()

#     def forward(self, x):
#         # print(f"inputsize{x.size()}")
#         x = self.linear1(x)
#         x = self.relu(x)
#         x = self.linear2(x)
#         return x

def create_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name is None:
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    else:
        raise NotImplementedError(f"{name} is not implemented.")

def create_norm(name):
    if name == "layernorm":
        return nn.LayerNorm
    elif name == "batchnorm":
        return nn.BatchNorm1d
    elif name == "graphnorm":
        return partial(NormLayer, norm_type="groupnorm")
    else:
        return None


class NormLayer(nn.Module):
    def __init__(self, hidden_dim, norm_type):
        super().__init__()
        if norm_type == "batchnorm":
            self.norm = nn.BatchNorm1d(hidden_dim)
        elif norm_type == "layernorm":
            self.norm = nn.LayerNorm(hidden_dim)
        elif norm_type == "graphnorm":
            self.norm = norm_type
            self.weight = nn.Parameter(torch.ones(hidden_dim))
            self.bias = nn.Parameter(torch.zeros(hidden_dim))

            self.mean_scale = nn.Parameter(torch.ones(hidden_dim))
        else:
            raise NotImplementedError
        
    def forward(self, graph, x):
        tensor = x
        if self.norm is not None and type(self.norm) != str:
            return self.norm(tensor)
        elif self.norm is None:
            return tensor

        batch_list = graph.batch_num_nodes
        batch_size = len(batch_list)
        batch_list = torch.Tensor(batch_list).long().to(tensor.device)
        batch_index = torch.arange(batch_size).to(tensor.device).repeat_interleave(batch_list)
        batch_index = batch_index.view((-1,) + (1,) * (tensor.dim() - 1)).expand_as(tensor)
        mean = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        mean = mean.scatter_add_(0, batch_index, tensor)
        mean = (mean.T / batch_list).T
        mean = mean.repeat_interleave(batch_list, dim=0)

        sub = tensor - mean * self.mean_scale

        std = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        std = std.scatter_add_(0, batch_index, sub.pow(2))
        std = ((std.T / batch_list).T + 1e-6).sqrt()
        std = std.repeat_interleave(batch_list, dim=0)
        return self.weight * sub / std + self.bias

class GCN(nn.Module):
    def __init__(self,
                 in_dim,
                 num_hidden,
                 out_dim,
                 num_layers,
                 dropout,
                 activation,
                 residual,
                 norm,
                 encoding=False
                 ):
        super(GCN, self).__init__()
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.gcn_layers = nn.ModuleList()
        self.activation = activation
        self.dropout = dropout
        last_activation = create_activation(activation) if encoding else None
        last_residual = encoding and residual
        last_norm = norm if encoding else None
        
        if num_layers == 1:
            self.gcn_layers.append(GraphConv(
                in_dim, out_dim, residual=last_residual, norm=last_norm, activation=last_activation))
        else:
            # input projection (no residual)
            self.gcn_layers.append(GraphConv(
                in_dim, num_hidden, residual=residual, norm=norm, activation=create_activation(activation)))
            # hidden layers
            for l in range(1, num_layers - 1):
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.gcn_layers.append(GraphConv(
                    num_hidden, num_hidden, residual=residual, norm=norm, activation=create_activation(activation)))
            # output projection
            self.gcn_layers.append(GraphConv(
                num_hidden, out_dim, residual=last_residual, activation=last_activation, norm=last_norm))

        if norm is not None:
            self.norms = nn.ModuleList([
                norm(num_hidden)
                for _ in range(num_layers - 1)
            ])
            if not encoding:
                self.norms.append(norm(out_dim))
        else:
            self.norms = None
        # self.norms = None
        self.head = nn.Identity()


    def forward(self, g, node_fea, edge_fea, return_hidden=False):
        h = node_fea
        k = edge_fea
        hidden_list = []
        for l in range(self.num_layers):
            h = F.dropout(h, p=self.dropout, training=self.training)
            h, k = self.gcn_layers[l](g, h, k)
            if self.norms is not None and l != self.num_layers - 1:
                h = self.norms[l](h)
            hidden_list.append(h)
        # output projection
        if self.norms is not None and len(self.norms) == self.num_layers:
            h = self.norms[-1](h)
        if return_hidden:
            return self.head(h), hidden_list
        else:
            return self.head(h)


def edge_message_fn(edges):#无聚合
    return {"e": torch.cat([edges.src['u'], edges.dst['u'], edges.data['e']], dim = 1)}    


class GraphConv(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 norm=None,
                 activation=None,
                 residual=True,
                 ):
        super().__init__()
        self._in_feats = in_dim
        self._out_feats = out_dim
        self.fc = nn.Linear(in_dim, out_dim)

        if residual:
            if self._in_feats != self._out_feats:
                self.res_fc = nn.Linear(
                    self._in_feats, self._out_feats, bias=False)
                print("! Linear Residual !")
            else:
                print("Identity Residual ")
                self.res_fc = nn.Identity()
        else:
            self.register_buffer('res_fc', None)

        if norm == "batchnorm":
            self.norm = nn.BatchNorm1d(out_dim)
        elif norm == "layernorm":
            self.norm = nn.LayerNorm(out_dim)
        else:
            self.norm = None
        
        self.norm = norm
        if norm is not None:
            self.norm = norm(out_dim)
        self._activation = activation

        self.reset_parameters()

    def reset_parameters(self):
        self.fc.reset_parameters()
    

    def comute_edge(self, g, node_fea):
        U, V = g.edges()
        u_buffer = []
        v_buffer = []
        for u, v in zip(U, V):
            u_buffer.append(node_fea[u])
            v_buffer.append(node_fea[v])
        u_tensor = torch.stack(u_buffer)
        v_tensor = torch.stack(v_buffer)
        edge_fea = L2_dist(u_tensor, v_tensor)
        return edge_fea


    def forward(self, graph, node_fea, edge_fea):
        print("gcov forward")
        with graph.local_scope():
            feat_src, feat_dst = expand_as_pair(node_fea, graph)#用于支持二分图
            degs = graph.out_degrees().float().clamp(min=1)#计算节点的出度矩阵
            print(len(degs))
            norm = torch.pow(degs, -0.5) #计算C_ij,归一化用
            shp = norm.shape + (1,) * (feat_src.dim() - 1)#resahpe
            norm = torch.reshape(norm, shp)
            try:
                feat_src = feat_src * norm.squeeze(0)
            except:
                print(len(degs))
                print(len(feat_src))
                print(len(norm))
                print(graph)
                assert False, "debug"
            # print(f"最一开始传入的大小{edge_fea.size()} {node_fea.size()}")
            graph.edata['e'] = edge_fea
            graph.srcdata['u'] = node_fea

            #首先对点的特征进行更新
            #先消息传递两次，并聚合,求取边和点的均值
            graph.update_all(fn.copy_u('u', 'temp'), fn.mean('temp', 'mean_u'))
            graph.update_all(fn.copy_e('e', 'temp'), fn.mean('temp', 'mean_e'))
            # print(f"sizes(): {graph.ndata['mean_u'].size()} {graph.ndata['mean_e'].size()}")
            graph.apply_nodes(lambda nodes: {'u': (nodes.data['mean_u'] + nodes.data['mean_e'] + nodes.data['u'])})

            #然后更新边，通过concat操作先聚合，然后再mlp映射
            # graph.apply_edges(edge_message_fn)
            # print(f"更新完边的特征是{graph.edata['e'].size()}")
            # temp = self.mlp(graph.edata['e'])
            #通过计算点的相似度更新边
            edge_fea_temp = self.comute_edge(graph, node_fea)

            # 确保a → b == b → a， 对a, b 取平均
            #u: [0, 1, 2,   3, 3, 3], v: [3, 3, 3,    0, 1, 2]
            # [1,2,3, 4,5,6]
            half_node_num = len(edge_fea_temp) // 2 
            # print(f"试试{edge_fea_temp.size()}, {half_node_num}")

            temp = (torch.cat([edge_fea_temp[:half_node_num], edge_fea_temp[:half_node_num]]) + torch.cat([edge_fea_temp[half_node_num:], edge_fea_temp[half_node_num:]])) / 2
            print(temp.size())
            flag = 0
            for i in range(len(temp)):
                if temp[i] != temp[half_node_num + i -1]:
                    print("hhh")
                    flag = 1
            if flag ==0:
                print("是的对称的")
            assert False
            #然后片段阈值
            # print(f"统计阈值信息 均值：{temp.mean()} 最大值{temp.max()}，最小值{temp.min()}")
            # qq = sorted(temp)
            # print(f"70%的值小于{qq[int(len(qq) * 0.7)]}")
            a = int(temp.mean())
            temp = torch.threshold(temp, a, 0) #TODO compute the threshold
            temp = temp.unsqueeze(1)
            graph.edata['e'] = temp
            rst = graph.dstdata['u']

            # print(f"after:{graph.edata}")
            # print(f"3:{graph}")
            rst = self.fc(rst)
            # print(f"1rst:{rst.size()}")
            # 更新函数
            # if self._norm in ['right', 'both']:
            degs = graph.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat_dst.dim() - 1)
            norm = torch.reshape(norm, shp)
            rst = rst * norm
            # print(f"2rst:{rst.size()}")
            if self.res_fc is not None:
                rst = rst + self.res_fc(feat_dst)

            if self.norm is not None:
                rst = self.norm(rst)

            if self._activation is not None:
                rst = self._activation(rst)
            # print(f"更新完的点和边的特征大小为{rst.size()}, {edge_fea.size()}")
            # print(f"size() {rst.size()}, {graph.edata['e'].size()}")
            return rst, graph.edata['e']

if __name__ == "__main__":
    # u, v = torch.tensor([0, 0, 0, 1]), torch.tensor([1, 2, 3, 3])
    # g = dgl.graph((u, v))   
    # bg = dgl.add_reverse_edges(g)
    # node_fea = torch.ones(4, 64)
    # edge_fea = torch.zeros(8, 64)
    # bg.ndata['u'] = node_fea
    # bg.edata['e'] = edge_fea
    # # print(bg.edges())

    # graph_model = GCN(in_dim=64, num_hidden=64, out_dim=64, num_layers=20, dropout=0,activation="prelu", residual=True,norm=nn.LayerNorm)
    # graph_model(bg, node_fea, edge_fea)
    # # # print(bg)
    # # # bg.apply_edges(edge_message_fn)
    # # bg.update_all(fn.copy_u('u', 'temp'), fn.mean('temp', 'mean_u'))
    # # bg.update_all(fn.copy_e('e', 'temp'), fn.mean('temp', 'mean_e'))
    # # bg.apply_nodes(lambda nodes: {'res': (nodes.data['mean_u'] + nodes.data['mean_e']) / 2})
    # # bg.apply_edges(edge_message_fn)
    # # print(bg.ndata)
    # print(bg.edata)

    m = torch.randn(32, 128)
    n = torch.randn(16, 128)
    a = L2_dist(m, n)
    print(a)
