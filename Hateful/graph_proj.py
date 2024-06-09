import dgl.nn.pytorch as dglnn
import torch.nn as nn
import torch.nn.functional as F
import dgl

class Classifier_proj(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(Classifier_proj, self).__init__()

        self.conv1 = dglnn.GraphConv(in_dim, hidden_dim,activation=F.elu)
        # self.layer1 = nn.Linear(in_dim, hidden_dim)
        # self.activation = nn.ELU()


        # self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim)
        #self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g, h):
        # 应用图卷积和激活函数
        h = self.conv1(g, h)
        # h = self.conv2(g, self.activation(h))
        
        # h = self.layer1(h)
        # h = self.activation(h)

        return h
        # h = F.relu(self.conv2(g, h))
        # with g.local_scope():
        #     g.ndata['h'] = h
        #     # 使用平均读出计算图表示
        #     hg = dgl.mean_nodes(g, 'h')
        #     return self.classify(hg)