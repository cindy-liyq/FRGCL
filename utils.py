import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn

seed = 2023
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)


# 从s-h图中得到s-s的相似性以及h-h的相似性
def getSim_from_sh(adj_oh):
    num_s = adj_oh.shape[0]
    num_h = adj_oh.shape[1]
    # 得到s-s的相似性
    # 计算第一行与第二行相等元素值的个数
    ss_sim = np.zeros((num_s, num_s))
    for s1 in range(num_s):
        for s2 in range(s1, num_s):
            equal_count = np.sum(adj_oh[s1] == adj_oh[s2])
            ss_sim[s1][s2] = equal_count / num_h
            ss_sim[s2][s1] = equal_count / num_h

    # 得到h-h的相似性
    hh_sim = np.zeros((num_h, num_h))
    for h1 in range(num_h):
        for h2 in range(h1, num_h):
            equal_count = np.sum(adj_oh[:, h1] == adj_oh[:, h2])
            hh_sim[h1][h2] = equal_count / num_s
            hh_sim[h2][h1] = equal_count / num_s

    return ss_sim, hh_sim


# normalizing the adj matrix
def normal_adj_matrix(coo_adj):
    rowD = np.array(coo_adj.sum(1)).squeeze()
    colD = np.array(coo_adj.sum(0)).squeeze()
    for i in range(len(coo_adj.data)):
        # coo_adj.data[i] = coo_adj.data[i] / pow(rowD[coo_adj.row[i]] * colD[coo_adj.col[i]], 0.5)
        coo_adj.data[i] = coo_adj.data[i] / pow(rowD[coo_adj.row[i]] * colD[coo_adj.col[i]], 0.5)

    return coo_adj


def get_correlation_coefficient(adj_oh):
    neighbors = {i: [] for i in range(len(adj_oh))}
    row, col = np.where(adj_oh != 0)
    start = 0
    k = 0
    for i, r in enumerate(row):
        end = r
        if start != end:
            neighbors[start] = col[k:i].tolist()
            start = end
            k = i
    return neighbors


def jaccard_similarity(neighbor1, neighbor2):
    intersection = len(set(neighbor1).intersection(neighbor2))
    union = len(set(neighbor1).union(neighbor2))
    if union == 0:
        return 0
    return intersection / union


def get_similarity(neighbors):
    # 创建一个空的相似性矩阵，用于存储节点之间的 Jaccard 相关性
    num_nodes = len(neighbors)
    similarity_matrix = np.zeros((num_nodes, num_nodes))

    # 计算每一对节点之间的 Jaccard 相关性
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            node1 = i
            node2 = j
            neighbor1 = neighbors[node1]
            neighbor2 = neighbors[node2]
            similarity = jaccard_similarity(neighbor1, neighbor2)
            similarity_matrix[node1, node2] = similarity
            similarity_matrix[node2, node1] = similarity

    # 打印节点之间的 Jaccard 相关性矩阵
    return similarity_matrix


def scipy_sparse_mat_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def sparse_dropout(mat, dropout):
    if dropout == 0.0:
        return mat
    indices = mat.indices()
    values = nn.functional.dropout(mat.values(), p=dropout)
    size = mat.size()
    return torch.sparse.FloatTensor(indices, values, size)


class TrnData(Dataset):
    def __init__(self, coomat):
        self.rows = coomat.row
        self.cols = coomat.col
        self.dokmat = coomat.todok()
        self.negs = np.zeros(len(self.rows)).astype(np.float32)

    def neg_sampling(self):
        for i in range(len(self.rows)):
            u = self.rows[i]
            while True:
                i_neg = np.random.randint(self.dokmat.shape[1])
                if (u, i_neg) not in self.dokmat:
                    break
            self.negs[i] = i_neg

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx], self.cols[idx], self.negs[idx]


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
