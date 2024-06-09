"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
from typing import List, Dict
import torch.nn as nn
import torch

__all__ = ['DomainDiscriminator']


class DomainDiscriminator(nn.Module):
    r"""Domain discriminator model from
    `Domain-Adversarial Training of Neural Networks (ICML 2015) <https://arxiv.org/abs/1505.07818>`_
    Distinguish whether the input features come from the source domain or the target domain.
    The source domain label is 1 and the target domain label is 0.
    Args:
        in_feature (int): dimension of the input feature
        hidden_size (int): dimension of the hidden features
        batch_norm (bool): whether use :class:`~torch.nn.BatchNorm1d`.
            Use :class:`~torch.nn.Dropout` if ``batch_norm`` is False. Default: True.
    Shape:
        - Inputs: (minibatch, `in_feature`)
        - Outputs: :math:`(minibatch, 1)`
    """

    def __init__(self, in_feature: int, hidden_size: int, batch_norm=True):
        super(DomainDiscriminator, self).__init__()
        if batch_norm :
            self.linears = nn.ModuleList([
                nn.Linear(in_feature, hidden_size), 
                nn.BatchNorm1d(hidden_size),   
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
                nn.Sigmoid()
            ])
                
        else :
            self.linears = nn.ModuleList([
                nn.Linear(in_feature, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(hidden_size, 1),
                nn.Sigmoid()
            ])
    def forward(self, h: torch.Tensor)-> torch.Tensor:
        h = self.linears[0](h)
        h = self.linears[1](h)
        h = self.linears[2](h)
        h = self.linears[3](h)
        h = self.linears[4](h)
        h = self.linears[5](h)
        h = self.linears[6](h)
        h = self.linears[7](h)
        return h

    def get_parameters(self) -> List[Dict]:
        return [{"params": self.parameters(), "lr": 1.}]