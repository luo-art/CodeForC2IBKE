import torch_sparse

from helper import *
from model.compgcn_conv_basis import CompGCNConvBasis
import torch.nn as nn
from torch.nn import functional as functional
from torch_cluster import random_walk

class BaseModel(torch.nn.Module):
    def __init__(self, params):
        super(BaseModel, self).__init__()

        self.p		= params
        self.act	= torch.tanh
        self.bceloss	= torch.nn.BCELoss()

    def loss(self, pred, true_label):
        return self.bceloss(pred, true_label)

# class CompGCNBase(BaseModel):
#     def __init__(self, edge_index, edge_type, num_rel, params=None):
#         super(CompGCNBase, self).__init__(params)
#
#         self.edge_index		= edge_index
#         self.edge_type		= edge_type
#         self.p.gcn_dim		= self.p.embed_dim if self.p.gcn_layer == 1 else self.p.gcn_dim
#         self.init_embed		= get_param((self.p.num_ent,   self.p.init_dim))
#         self.device		= self.edge_index.device
#
#         if self.p.num_bases > 0:
#             self.init_rel  = get_param((self.p.num_bases,   self.p.init_dim))
#         else:
#             if self.p.score_func == 'transe': 	self.init_rel = get_param((num_rel,   self.p.init_dim))
#             else: 					self.init_rel = get_param((num_rel*2, self.p.init_dim))
#
#         if self.p.num_bases > 0:
#             self.conv1 = CompGCNConvBasis(self.p.init_dim, self.p.gcn_dim, num_rel, self.p.num_bases, act=self.act, params=self.p)
#             self.conv2 = CompGCNConv(self.p.gcn_dim,    self.p.embed_dim,    num_rel, act=self.act, params=self.p) if self.p.gcn_layer == 2 else None
#         else:
#             self.conv1 = CompGCNConv(self.p.init_dim, self.p.gcn_dim,      num_rel, act=self.act, params=self.p)
#             self.conv2 = CompGCNConv(self.p.gcn_dim,    self.p.embed_dim,    num_rel, act=self.act, params=self.p) if self.p.gcn_layer == 2 else None
#
#         self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent)))
#
#     def forward_base(self, sub, rel, drop1, drop2):
#
#         r	= self.init_rel if self.p.score_func != 'transe' else torch.cat([self.init_rel, -self.init_rel], dim=0)
#         x, r	= self.conv1(self.init_embed, self.edge_index, self.edge_type, rel_embed=r)
#         x	= drop1(x)
#         x, r	= self.conv2(x, self.edge_index, self.edge_type, rel_embed=r) 	if self.p.gcn_layer == 2 else (x, r)
#         x	= drop2(x) 							if self.p.gcn_layer == 2 else x
#
#         sub_emb	= torch.index_select(x, 0, sub)
#         rel_emb	= torch.index_select(r, 0, rel)
#
#         return sub_emb, rel_emb, x

class Rescal(BaseModel):
    def __init__(self, edge_index, edge_type, num_rel, params=None):
        super(Rescal, self).__init__(params)

class C2IBKE(Rescal):
    def __init__(self, edge_index, edge_type,adj, params=None):
        super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params)
        self.p = params
        self.entity_embeddings = nn.Parameter(torch.Tensor(self.p.num_ent, 200))
        nn.init.xavier_uniform_(self.entity_embeddings, gain=nn.init.calculate_gain('relu'))
        # 初始化向量
        self.relation_embeddings = nn.Parameter(torch.Tensor(self.p.num_rel * 2, 200 * 200))
        nn.init.xavier_uniform_(self.relation_embeddings, gain=nn.init.calculate_gain('relu'))
        self.register_parameter('b', torch.nn.Parameter(torch.zeros(self.p.num_ent)))
        self.lhs = self.entity_embeddings
        self.rel = self.relation_embeddings
        self.rhs = self.entity_embeddings
        self.walk_length = 3
        self.entity_num = self.p.num_ent
        self.device = torch.device('cuda')
        self.edge_index = edge_index
        self.cl2 = Contrast(200,0.6,0.5).cuda()
        self.create_sparse_adjaceny()
        self.node_mask_learner = nn.ModuleList([nn.Sequential(nn.Linear(200, 200), nn.ReLU(),
                                                              nn.Linear(200, 1)) for i in range(2)])
        self.edge_mask_learner = nn.ModuleList([nn.Sequential(nn.Linear(2*200, 200), nn.ReLU(),
                                                              nn.Linear(200, 1)) for i in range(2)])
    # 创建度矩阵
    def create_sparse_adjaceny(self):
        index = [self.edge_index[0].tolist(), self.edge_index[1].tolist()]
        value = [1.0] * len(self.edge_index[0])
        print(len(value))
        # user_num * item_num
        # 利用三元组的形式创建稀疏矩阵，即用户和商品的交互矩阵
        self.interact_matrix = torch.sparse_coo_tensor(index, value, (self.entity_num, self.entity_num))
        tmp_index = [self.edge_index[0].tolist(), self.edge_index[1].tolist()]
        tmp_adj = torch.sparse_coo_tensor(tmp_index, value,
                                          (self.entity_num, self.entity_num))

        # 邻接矩阵
        self.joint_adjaceny_matrix = (tmp_adj)

        # 计算每个节点的度，并且进行归一化，形成度矩阵
        degree = torch.sparse.sum(self.joint_adjaceny_matrix, dim=1).to_dense()
        degree = torch.pow(degree, -0.5)
        degree[torch.isinf(degree)] = 0
        D_inverse = torch.diag(degree, diagonal=0).to_sparse()
        self.joint_adjaceny_matrix_normal = torch.sparse.mm(torch.sparse.mm(D_inverse, self.joint_adjaceny_matrix),
                                                            D_inverse)
        joint_indices = self.joint_adjaceny_matrix_normal.indices()
        self.row = joint_indices[0]
        self.col = joint_indices[1]
        start = torch.arange(self.entity_num)
        walk = random_walk(self.row, self.col, start, walk_length=self.walk_length)
        self.rw_adj = torch.zeros((self.entity_num, self.entity_num))
        self.rw_adj = torch.scatter(self.rw_adj, 1, walk, 1).to_sparse()
        degree = torch.sparse.sum(self.rw_adj, dim=1).to_dense()
        degree = torch.pow(degree, -1)
        degree[torch.isinf(degree)] = 0
        D_inverse = torch.diag(degree, diagonal=0).to_sparse()
        # user_num+item_num * user_num+item_num
        self.rw_adj = torch.sparse.mm(D_inverse, self.rw_adj).to(self.device)
        self.joint_adjaceny_matrix_normal = self.joint_adjaceny_matrix_normal.to(self.device)

    # 小技巧:重参数化
    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        for modu in self.node_mask_learner:
            for model in modu:
                if isinstance(model, nn.Linear):
                    nn.init.xavier_normal_(model.weight, gain)

    # 计算互信息损失
    def ssl_compute(self, normalized_embedded_s1, normalized_embedded_s2):
        pos_score = torch.sum(torch.mul(normalized_embedded_s1, normalized_embedded_s2), dim=1, keepdim=False)
        all_score = torch.mm(normalized_embedded_s1, normalized_embedded_s2.t())
        ssl_mi = torch.log(torch.exp(pos_score / 0.9) / torch.exp(all_score / 0.9).sum(dim=1,
                                                                                       keepdim=False)).mean()
        return -ssl_mi


    def forward(self, sub, rel,obj,neg_tail,neg=None,runmode = None,mod = 0):
        node_mask_list = []
        edge_mask_list = []
        cur_embedding = self.lhs
        cur_embedding_node_drop = cur_embedding
        cur_embedding_edge_drop = cur_embedding
        # 这一段代码的工作是:
        # 决定删除哪些节点和哪些边
        for i in range(2):
            node_mask = self.node_mask_learner[i](cur_embedding)
            bias = 0.0 + 0.0001  # If bias is 0, we run into problems
            eps = (bias - (1 - bias)) * torch.rand(node_mask.size()) + (1 - bias)
            node_gate_inputs = torch.log(eps) - torch.log(1 - eps)
            node_gate_inputs = node_gate_inputs.to(self.device)
            node_gate_inputs = (node_gate_inputs + node_mask) / 0.5
            node_mask = torch.sigmoid(node_gate_inputs)
            node_mask_list.append(node_mask)

            edge_cat_embedding = torch.cat([cur_embedding[self.row], cur_embedding[self.col]], dim=-1)
            edge_mask = self.edge_mask_learner[i](edge_cat_embedding)  # (951118,1)
            bias = 0.0 + 0.0001  # If bias is 0, we run into problems
            eps = (bias - (1 - bias)) * torch.rand(edge_mask.size()) + (1 - bias)
            edge_gate_inputs = torch.log(eps) - torch.log(1 - eps)
            edge_gate_inputs = edge_gate_inputs.to(self.device)
            edge_gate_inputs = (edge_gate_inputs + edge_mask) / 0.5
            edge_mask = torch.sigmoid(edge_gate_inputs).squeeze(1)
            edge_mask_list.append(edge_mask)

        for i in range(2):
            node_mask = node_mask_list[i]
            mean_pooling_embedding = torch.mm(self.rw_adj, cur_embedding_node_drop)
            cur_embedding_node_drop = torch.mul(node_mask, cur_embedding_node_drop) + torch.mul((1 - node_mask),
                                                                                            mean_pooling_embedding)
            cur_embedding_node_drop = torch.mm(self.joint_adjaceny_matrix_normal, cur_embedding_node_drop)
        user_embeddings_node_drop = cur_embedding_node_drop

        for i in range(2):
            edge_mask = edge_mask_list[i]
            cur_embedding_edge_drop = torch.mm(self.rw_adj, cur_embedding_edge_drop)
            new_edge = torch.mul(self.joint_adjaceny_matrix_normal.values(), edge_mask)
            # 这里的计算原理是什么？？？反正得到的是去除边的节点更新表示
        cur_embedding_edge_drop = torch_sparse.spmm(self.joint_adjaceny_matrix_normal.indices(), new_edge, self.entity_num,self.entity_num, cur_embedding_edge_drop)
        user_embeddings_edge_drop = cur_embedding_edge_drop
        #计算entrocrossLoss
        lhs1 = torch.index_select(cur_embedding, 0, sub)
        rel1 = torch.index_select(self.rel, 0, rel).reshape(-1, 200, 200)
        self_hr = (torch.bmm(lhs1.unsqueeze(1), rel1)).squeeze()
        rhs1 = torch.index_select(cur_embedding,0,obj)
        x = self_hr @ cur_embedding.t()
        tmp = x
        x = torch.sigmoid(x)
        cl2_loss=0
        cl2_loss1=0
        contrast_loss = 0
        # caculate the ssl_compute
        if mod == 1:
            h = torch.index_select(cur_embedding, 0, neg_tail[:, 0])
            h1 = torch.index_select(cur_embedding_node_drop, 0, neg_tail[:, 0])
            h2 = torch.index_select(cur_embedding_edge_drop, 0, neg_tail[:, 0])
            r = torch.index_select(self.rel, 0, neg_tail[:, 1]).reshape(-1, 200, 200)
            hr = (torch.bmm(h.unsqueeze(1), r)).squeeze()
            hr1 = (torch.bmm(h1.unsqueeze(1), r)).squeeze()
            hr2 = (torch.bmm(h2.unsqueeze(1), r)).squeeze()
            contrast_loss = self.cl2(self_hr,hr,labels1 = sub,labels2 = rel)
            cl2_loss = self.cl2(self_hr,hr1,labels1=sub, labels2=rel)
            cl2_loss1 = self.cl2(self_hr,hr2, labels1=sub, labels2=rel)
        # 计算原视角和IB图视角下的互信息损失
        # user_embedding = cur_embedding[torch.unique(sub)]
        # normalized_user_embedded_unique = functional.normalize(user_embedding)
        # user_embedded_node_drop = user_embeddings_node_drop[torch.unique(sub)]
        # normalized_user_embedded_node_drop = functional.normalize(user_embedded_node_drop)
        # ssl_node = self.ssl_compute(normalized_user_embedded_node_drop,normalized_user_embedded_unique)
        # user_embedded_edge_drop = user_embeddings_edge_drop[torch.unique(sub)]
        # normalized_user_embedded_edge_drop = functional.normalize(user_embedded_edge_drop)
        # ssl_edge = self.ssl_compute(normalized_user_embedded_edge_drop,normalized_user_embedded_unique)
        if runmode == "train":
            tmp = torch.stack([tmp[i].index_select(0, neg[i]) for i in range(tmp.shape[0])], 0)
            x = torch.stack([x[i].index_select(0, neg[i]) for i in range(x.shape[0])], 0)
        return x,self.p.lamda * (cl2_loss1 + cl2_loss)+ self.p.beta * contrast_loss,[(lhs1,rel1,rhs1)],tmp

class Contrast(nn.Module):

    def __init__(self, hidden_dim, tau, lambda_):
        """对比损失模块

        :param hidden_dim: int 隐含特征维数
        :param tau: float 温度参数
        :param lambda_: float 0~1之间，网络结构视图损失的系数（元路径视图损失的系数为1-λ）
        """
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.elu=nn.ELU()
        self.linear=torch.nn.Linear(200,200)


        self.tau = tau
        self.lambda_ = lambda_
        self.reset_parameters()
    def get_negative_mask(self, batch_size, labels1=None, labels2=None):
        if labels2 is None:
            labels1 = labels1.contiguous().view(-1, 1)
            if labels1.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels1, labels1.T).float().cuda()
        else:
            labels1 = labels1.contiguous().view(-1, 1)#contiguous()用于确保张量在内存中是连续存储的
            mask1 = torch.eq(labels1, labels1.T).float().cuda()#torch.eq() 是一个逐元素比较函数，用于比较两个张量的对应元素是否相等
            labels2 = labels2.contiguous().view(-1, 1)
            mask2 = torch.eq(labels2, labels2.T).float().cuda()
            mask = mask1*mask2
            mask = mask.float().cuda()
        return mask

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        for model in self.proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain)

    #计算互信息(基于正样本之间的相似度)得到对比学习的loss
    def ssl_compute(self, x, y):
        """计算相似度矩阵

        :param x: tensor(N, d)
        :param y: tensor(N, d)
        :return: tensor(N, N) S[i, j] = exp(cos(x[i], y[j]))
        """
        x_norm = torch.norm(x, dim=1, keepdim=True)
        y_norm = torch.norm(y, dim=1, keepdim=True)
        semi = torch.mm(x,y.t())
        denominator = torch.mm(x_norm, y_norm.t())
        return torch.exp(semi/denominator / 0.6)

    def forward(self, z_sc, z_mp, labels1=None,labels2=None):
        """
        :param z_sc: tensor(N, d) 目标顶点在网络结构视图下的嵌入
        :param z_mp: tensor(N, d) 目标顶点在元路径视图下的嵌入
        :param pos: tensor(B, N) 0-1张量，每个目标顶点的正样本
            （B是batch大小，真正的目标顶点；N是B个目标顶点加上其正样本后的顶点数）
        :return: float 对比损失

        """
        pos = self.get_negative_mask(z_sc.shape[0], labels1, labels2).cuda()
        z_sc_proj = self.proj(z_sc)
        z_mp_proj = self.proj(z_mp)
        sim_sc2mp = self.ssl_compute(z_sc_proj, z_mp_proj)
        sim_mp2sc = sim_sc2mp.t()
        batch = pos.shape[0]
        sim_sc2mp = sim_sc2mp / (sim_sc2mp.sum(dim=1, keepdim=True) + 1e-8)
        loss_sc = -torch.log(torch.sum(sim_sc2mp[:batch] * pos, dim=1)).mean()
        sim_mp2sc = sim_mp2sc / (sim_mp2sc.sum(dim=1, keepdim=True) + 1e-8)
        loss_mp = -torch.log(torch.sum(sim_mp2sc[:batch] * pos, dim=1)).mean()
        return self.lambda_ * loss_sc + (1 - self.lambda_) * loss_mp