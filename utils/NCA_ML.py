import torch
from torch import nn
from torch.autograd import Function
import math

eps = 1e-8

class NCA_ML_CrossEntropy(nn.Module):
    ''' \sum log(w_{ij}*p_{ij})
        Store all the labels of the dataset.
        Only pass the indexes of the training instances during forward. 
    '''

    def __init__(self, multiHotMtx, margin=0):
        super().__init__()

        self.register_buffer('multiHotMtx', multiHotMtx)
        # transfer 0,1 multihot to -1,1
        multiHotMtx[multiHotMtx==0] = -1
        self.multiHotMtx = multiHotMtx
        self.labelNum = self.multiHotMtx.size(1)
        self.margin = margin

    def forward(self, x, indexes):
        
        batchSize = x.size(0)
        n = x.size(1)
        exp = torch.exp(x)

        batch_multiHotMtx = torch.index_select(self.multiHotMtx, 0, indexes.data)

        out = torch.mm(batch_multiHotMtx, self.multiHotMtx.t())
        hamming_dist = (out + self.labelNum) / 2
        weights = hamming_dist / self.labelNum
        
        # print(weights.max(), weights.min())


        # self prob exclusion, hack with memory for effeciency
        exp.data.scatter_(1, indexes.data.view(-1,1), 0)

        p = torch.mul(exp, weights.float()).sum(dim=1)
        Z = exp.sum(dim=1)

        Z_exclude = Z - p
        p = p.div(math.exp(self.margin))
        Z = Z_exclude + p

        prob = torch.div(p, Z)
        prob_masked = torch.masked_select(prob, prob.ne(0))

        loss = prob_masked.log().sum(0)

        return - loss / batchSize
























