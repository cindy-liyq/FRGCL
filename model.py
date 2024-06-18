import torch
import torch.nn as nn
from utils import sparse_dropout
import numpy as np
import torch.nn.functional as F

seed = 2023
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)
class LightGCL(nn.Module):
    def __init__(self, n_s, n_h, dim, u_mul_s, v_mul_s, ut, vt, train_csr, adj_norm,ss_adj_norm,hh_adj_norm, layer, temp, lambda_1, lambda_2,
                 dropout, batch_user, device):
        super(LightGCL, self).__init__()
        self.E_s_0 = nn.Parameter(nn.init.xavier_normal_(torch.empty(n_s, dim)))
        self.E_h_0 = nn.Parameter(nn.init.xavier_normal_(torch.empty(n_h, dim)))

        self.E_ss_0 = nn.Parameter(nn.init.xavier_normal_(torch.empty(n_s, dim)))########################加s-s
        self.E_hh_0 = nn.Parameter(nn.init.xavier_normal_(torch.empty(n_h, dim)))########################加h-h

        self.train_csr = train_csr
        self.adj_norm = adj_norm
        self.ss_adj_norm = ss_adj_norm##############################加s-s
        self.hh_adj_norm = hh_adj_norm##############################加h-h
        self.layer = layer
        self.E_s_list = [None] * (layer + 1)
        self.E_h_list = [None] * (layer + 1)
        self.E_ss_list = [None] * (layer + 1)########################加s-s
        self.E_hh_list = [None] * (layer + 1)########################加h-h
        self.E_s_list[0] = self.E_s_0
        self.E_h_list[0] = self.E_h_0
        self.E_ss_list[0] = self.E_ss_0########################加s-s
        self.E_hh_list[0] = self.E_hh_0########################加h-h
        self.Z_s_list = [None] * (layer + 1)  # Z_s_list:聚合sym的每一层特征
        self.Z_h_list = [None] * (layer + 1)  # Z_h_list:聚合herb的每一层特征
        self.Z_ss_list = [None] * (layer + 1)  # Z_ss_list:聚合sym的每一层特征 ########################加s-s
        self.Z_hh_list = [None] * (layer + 1)  # Z_hh_list:聚合herb的每一层特征 ########################加h-h
        self.G_s_list = [None] * (layer + 1)
        self.G_h_list = [None] * (layer + 1)
        self.G_ss_list = [None] * (layer + 1)########################加s-s
        self.G_hh_list = [None] * (layer + 1)########################加h-h
        self.G_s_list[0] = self.E_s_0
        self.G_h_list[0] = self.E_h_0
        self.G_ss_list[0] = self.E_ss_0########################加s-s
        self.G_hh_list[0] = self.E_hh_0########################加h-h
        self.temp = temp
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.dropout = dropout
        self.act = nn.LeakyReLU(0.5)
        self.batch_user = batch_user

        self.E_s = None
        self.E_h = None

        self.u_mul_s = u_mul_s
        self.v_mul_s = v_mul_s
        self.ut = ut
        self.vt = vt
        self.dim = dim

        # self.mlp = torch.nn.Linear(dim, dim)
        self.SI_bn = torch.nn.BatchNorm1d(dim)
        self.relu = torch.nn.ReLU()


        self.device = device


    def forward(self, sids, hids, pos, neg,ps, test=False):
        if test==True:
            e_synd = torch.mm(ps, self.E_s+self.E_ss)  # (512,64)--->512*390 @ 390*64  # prescription * es########################加s-s
            # batch*1
            preSum = ps.sum(dim=1).view(-1, 1)  # (512,1)
            # batch*dim
            e_synd_norm = e_synd / preSum  # (512,256)
            e_synd_norm = self.SI_bn(e_synd_norm)  # (512,256)
            e_synd_norm = self.relu(e_synd_norm)  # (512,256)  # batch*dim
            pre = torch.mm(e_synd_norm, self.E_h.t()+self.E_hh.t())  # (512,805)--->512*256 @ 256*805##########################加h-h
            return pre
        else:
            for layer in range(1, self.layer + 1):  # 2层GCN
                # GNN propagation
                # 第0层时随机初始化得到的
                # 第二层是通过拉普拉斯矩阵乘法---标准化后的邻接矩阵与第0层进行矩阵乘法得到的
                self.Z_s_list[layer] = (torch.spmm(sparse_dropout(self.adj_norm, self.dropout), self.E_h_list[layer - 1]))
                self.Z_h_list[layer] = (
                    torch.spmm(sparse_dropout(self.adj_norm, self.dropout).transpose(0, 1), self.E_s_list[layer - 1]))

                self.Z_ss_list[layer] = (
                    torch.spmm(sparse_dropout(self.ss_adj_norm, self.dropout), self.E_ss_list[layer - 1]))########################加s-s

                self.Z_hh_list[layer] = (
                    torch.spmm(sparse_dropout(self.hh_adj_norm, self.dropout),
                               self.E_hh_list[layer - 1]))  ########################加h-h

                # svd_adj propagation
                vt_eh = self.vt @ self.E_h_list[layer - 1] #(5,753) @ (753,256)  ---->(5,256)
                self.G_s_list[layer] = (self.u_mul_s @ vt_eh) #(360,5) @ (5,256) ----->(360,256)
                ut_es = self.ut @ self.E_s_list[layer - 1] #(5,360) @ (360,256)  ---->(5,256)
                self.G_h_list[layer] = (self.v_mul_s @ ut_es)#(753,5) @ (5,256) ----->(753,256)



                # aggregate
                self.E_s_list[layer] = self.Z_s_list[layer]  # 经过一层聚合后的特征作为新的特征
                self.E_h_list[layer] = self.Z_h_list[layer]
                self.E_ss_list[layer] = self.Z_ss_list[layer]########################加s-s
                self.E_hh_list[layer] = self.Z_hh_list[layer]########################加h-h


            self.G_s = sum(self.G_s_list)
            self.G_h = sum(self.G_h_list)

            # aggregate across layers
            self.E_s = sum(self.E_s_list)
            self.E_h = sum(self.E_h_list)

            self.E_ss = sum(self.E_ss_list)########################加s-s
            self.E_hh = sum(self.E_hh_list)########################加h-h
            #self.E_s += self.E_ss########################加s-s    #############改动

            e_synd = torch.mm(ps, self.E_s+self.E_ss)  # (512,64)--->512*390 @ 390*64  # prescription * es  ##########################加s-s
            # batch*1
            preSum = ps.sum(dim=1).view(-1, 1)  # (512,1)
            # batch*dim
            e_synd_norm = e_synd / preSum  # (512,256)
            # e_synd_norm = self.mlp(e_synd_norm)  # (512,1,256)加个mlp层
            # e_synd_norm = e_synd_norm.view(-1, 64)  # (512,256)
            e_synd_norm = self.SI_bn(e_synd_norm)  # (512,256)
            e_synd_norm = self.relu(e_synd_norm)  # (512,256)  # batch*dim
            pre = torch.mm(e_synd_norm, self.E_h.t()+self.E_hh.t())  # (512,805)--->512*256 @ 256*805  ##########################加h-h
            # softmax_result = np.exp(pre.cpu().detach().numpy()) / np.sum(np.exp(pre.cpu().detach().numpy()), axis=1, keepdims=True)
            # softmax_result = F.softmax(pre, dim=1)


            # 计算损失
            # cl loss 对比损失，目的是使正样本更接近，负样本更分散
            G_s_norm = self.G_s
            E_s_norm = self.E_s
            G_h_norm = self.G_h
            E_h_norm = self.E_h


            neg_score = torch.log(torch.exp(G_s_norm[sids] @ E_s_norm.T / self.temp).sum(
                1) + 1e-8).mean()  # 负样本得分，通过计算负样本与输入样本之间的相似性分数，并将这些分数转换为概率分布，然后取对数。最后取平均得到负样本得分。
            neg_score += torch.log(torch.exp(G_h_norm[hids] @ E_h_norm.T / self.temp).sum(1) + 1e-8).mean()
            pos_score = (torch.clamp((G_s_norm[sids] * E_s_norm[sids]).sum(1) / self.temp, -5.0, 5.0)).mean() + (
                torch.clamp((G_h_norm[hids] * E_h_norm[hids]).sum(1) / self.temp, -5.0,
                            5.0)).mean()  # 正样本得分，通过计算正样本与输入样本之间的相似性分数，并将结果缩放到区间[-5.0, 5.0]内，然后取平均得到正样本得分。

            loss_s = -pos_score + neg_score

            # bpr loss  对于每个用户，我们选择一个正样本项目（用户已经有过交互行为的项目）和一个负样本项目（用户没有进行过交互的项目）。然后，通过计算用户和项目之间的相似性得分（通常使用内积或其他相似性度量），BPR损失的目标是最大化用户与正样本项目之间的相似性，同时最小化用户与负样本项目之间的相似性。
            s_emb = self.E_s[sids]
            pos_emb = self.E_h[pos]
            neg_emb = self.E_h[neg]
            pos_scores = (s_emb * pos_emb).sum(-1)
            neg_scores = (s_emb * neg_emb).sum(-1)
            loss_r = -(pos_scores - neg_scores).sigmoid().log().mean()

            # reg loss
            loss_reg = 0
            for param in self.parameters():
                loss_reg += param.norm(2).square()
            loss_reg *= self.lambda_2

            # total loss
            loss = loss_r + self.lambda_1 * loss_s + loss_reg
            return loss, loss_r, self.lambda_1 * loss_s,pre