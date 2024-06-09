from module.rela_encoder import Rela_Module
import torch.nn as nn
import torch
from torch.nn import functional
from module.classifier_latent import SingleClassifier,SingleClassifier_H, SingleClassifier_T

def convert_to_one_hot(indices, num_classes):
    batch_size = indices.size(0)
    indices = indices.unsqueeze(1)
    one_hot = indices.new_zeros(batch_size, num_classes).scatter_(1, indices, 1).cuda()
    return one_hot

def masked_softmax(logits, mask=None):
    eps = 1e-20
    probs = functional.softmax(logits, dim=1)
    if mask is not None:
        mask = mask.float()
        probs = probs * mask + eps
        probs = probs / probs.sum(1, keepdim=True)
    return probs

def st_gumbel_softmax(logits, temperature=1.0, mask=None):
    eps = 1e-20
    u = logits.data.new(*logits.size()).uniform_()   #返回一个与self张量具有相同数据类型和设备的空张量   从Gumbel(0, 1)分布中抽取的独立同分布样本
    gumbel_noise = -torch.log(-torch.log(u + eps) + eps)  #只有加的噪声是gumbel分布，最后的概率分布才跟原来的分布差不多，加高斯分布和均匀分布的噪声的概率分布跟原来的概率分布明显差别很大
    y = logits + gumbel_noise
    y = masked_softmax(logits = y / temperature, mask=mask)  #我们使用softmax函数来作为argmax的一个连续、可微的近似，并且生成维的样本向量
    y_argmax = y.max(1)[1]
    y_hard = convert_to_one_hot(indices=y_argmax, num_classes=y.size(1)).float()
    y = (y_hard - y).detach() + y
    return y

class Latent_Bert(nn.Module):
    def __init__(self):
        super(Latent_Bert,self).__init__()
        self.transfomer=Rela_Module(768, 768, 8, 512, 1, 0.1)
        self.softmax=nn.Softmax(dim=1)
        self.temp=0.3
        self.fc_gt=SingleClassifier(768,6,0.1)
        self.fc_a=SingleClassifier(768,6,0.1)
        
    def forward(self, image, text):
        align_repre=self.transfomer(image, text)
        text = text[:,0,:]
        dis_image = self.softmax(self.fc_a(align_repre))
        dis_text = st_gumbel_softmax(self.softmax(self.fc_gt(text)),self.temp)
        return dis_image, dis_text


class Latent_Bert_H(nn.Module):
    def __init__(self):
        super(Latent_Bert_H,self).__init__()
        self.transfomer=Rela_Module(768, 768, 8, 512, 1, 0.1)
        self.softmax=nn.Softmax(dim=1)
        self.temp=0.3
        self.fc_gt=SingleClassifier_H(768,20,0.2)
        self.fc_a=SingleClassifier_H(768,20,0.2)
        
    def forward(self, image, text):
        align_repre=self.transfomer(image, text)
        text = text[:,0,:]
        #image = image[:,0,:]

        dis_image = self.softmax(self.fc_a(align_repre))
        dis_text = st_gumbel_softmax(self.softmax(self.fc_gt(text)),self.temp)
        return dis_image, dis_text

class Latent_Bert_T(nn.Module):
    def __init__(self):
        super(Latent_Bert_T,self).__init__()
        self.transfomer=Rela_Module(768, 768, 8, 512, 1, 0.1)
        self.softmax=nn.Softmax(dim=1)
        self.temp=0.3
        self.fc_gt=SingleClassifier_T(768,20,0.2)
        self.fc_a=SingleClassifier_T(768,20,0.2)
        
    def forward(self, image, text):
        align_repre=self.transfomer(image, text)
        text = text[:,0,:]
        #image = image[:,0,:]

        dis_image = self.softmax(self.fc_a(align_repre))
        dis_text = st_gumbel_softmax(self.softmax(self.fc_gt(text)),self.temp)
        return dis_image, dis_text
