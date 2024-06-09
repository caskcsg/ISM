import dgl.nn.pytorch as dglnn
import torch.nn as nn
import torch.nn.functional as F
import dgl

class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim):  ##256  ###128
        super(Classifier, self).__init__()
        self.classify_1 = nn.Linear(in_dim, hidden_dim)
        self.classify_2 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
    def forward(self, g, h):
        # 使用平均读出计算图表示
        with g.local_scope():
            g.ndata['x'] = h
            hg = dgl.mean_nodes(g, 'x')
            hg = self.dropout(self.act(self.classify_1(hg)))
            hg = self.dropout(self.classify_2(hg))
            return hg
       
        