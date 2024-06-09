import torch.nn as nn
import torch


class CMD(nn.Module):
    """
    Adapted from https://github.com/wzell/cmd/blob/master/models/domain_regularizer.py
    """

    def __init__(self, trade_off=0.1):
        super(CMD, self).__init__()
        self.trade_off = trade_off

    def forward(self, f_s, f_t):
        U_s, _, _ = torch.svd(f_s.t())
        U_t, _, _ = torch.svd(f_t.t())
        P_s, cosine, P_t = torch.svd(torch.mm(U_s.t(), U_t))
        sine = torch.sqrt(1 - torch.pow(cosine, 2))
        rsd = torch.norm(sine, 1)  # Representation Subspace Distance
        bmp = torch.norm(torch.abs(P_s) - torch.abs(P_t), 2)  # Base Mismatch Penalization
        return rsd + self.trade_off * bmp