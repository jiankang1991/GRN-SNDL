
import numpy as np
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable, Function

from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, hamming_loss
from sklearn.metrics import f1_score, classification_report

class MetricTracker(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



class KNNClassification(nn.Module):
    """ 
    KNN also support multi-label classification
    """
    def __init__(self, X_train, Y_true, K=10):
        super().__init__()

        self.K = K

        self.KNN = KNeighborsClassifier(n_neighbors=self.K, weights='distance')
        self.KNN.fit(X_train, Y_true)

    def forward(self, X_test, y_true):

        y_pred = self.KNN.predict(X_test)
        
        # acc = accuracy_score(y_true, y_pred)
        # hammingLoss = hamming_loss(y_true, y_pred)
        sample_f1 = f1_score(y_true, y_pred, average="samples")

        return sample_f1


class calssification_report(nn.Module):

    def __init__(self, target_names):
        super().__init__()
        self.target_names = target_names
    def forward(self, predict_labels, true_labels):

        report = classification_report(true_labels, predict_labels, target_names=self.target_names, output_dict=True)

        return report

class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()



class HingeLoss(nn.Module):
    """
    Hinge loss based on the paper:
    when deep learning meets metric learning:remote sensing image scene classification
    via learning discriminative CNNs 
    https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/9
    """

    def __init__(self, margin=0.44):
        super().__init__()
        
        self.margin = margin

    def forward(self, oneHotCodes, features):
        
        L_S = oneHotCodes.mm(torch.t(oneHotCodes))
        Dist = torch.norm(features[:,None] - features, dim=2, p=2)**2

        Dist = self.margin - Dist
        
        L_S[L_S==0] = -1

        Dist = 0.05 - L_S * Dist

        loss = torch.triu(Dist, diagonal=1)

        loss[loss < 0] = 0

        return torch.mean(loss)




class F1_score(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, predict_labels, true_labels):

        macro_f1 = f1_score(true_labels, predict_labels, average="macro")
        micro_f1 = f1_score(true_labels, predict_labels, average="micro")
        sample_f1 = f1_score(true_labels, predict_labels, average="samples")

        return macro_f1, micro_f1, sample_f1



class ContrastiveLoss(nn.Module):
    """
    https://github.com/adambielski/siamese-triplet
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super().__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()


##############################
# Modified WARP loss utility 
# - Reference: https://arxiv.org/pdf/1312.4894.pdf
# https://github.com/Mipanox/Bird_cocktail/blob/master/utils.py

class WARP(Function): 
    """
    Autograd function of WARP loss. Appropirate for multi-label
    - Reference: 
      https://medium.com/@gabrieltseng/intro-to-warp-loss-automatic-differentiation-and-pytorch-b6aa5083187a
    """
    @staticmethod
    def forward(ctx, input, target, max_num_trials = None):
        batch_size = target.size()[0]
        label_size = target.size()[1]

        ## rank weight 
        rank_weights = [1.0/1]
        for i in range(1, label_size):
            rank_weights.append(rank_weights[i-1] + (1.0/i+1))

        if max_num_trials is None: 
            max_num_trials = target.size()[1] - 1

        ##
        positive_indices = target.gt(0).float()
        negative_indices = target.eq(0).float()
        L = torch.zeros(input.size())

        for i in range(batch_size):
            for j in range(label_size):
                if target[i,j] == 1:
                    ## initialization
                    sample_score_margin = -1
                    num_trials = 0

                    while ((sample_score_margin < 0) and (num_trials < max_num_trials)):
                        ## sample a negative label, to only determine L (ranking weight)
                        neg_labels_idx = np.array([idx for idx, v in enumerate(target[i,:]) if v == 0])

                        if len(neg_labels_idx) > 0:                        
                            neg_idx = np.random.choice(neg_labels_idx, replace=False)
                            ## if model thinks neg ranks before pos...
                            sample_score_margin = input[i,neg_idx] - input[i,j]
                            num_trials += 1

                        else: # ignore cases where all labels are 1...
                            num_trials = 1
                            pass

                    ## how many trials determine the weight
                    r_j = int(np.floor(max_num_trials / num_trials))
                    L[i,j] = rank_weights[r_j]
        
        ## summing over all negatives and positives
        #-- since inputs are sigmoided, no need for clamp with min=0
        loss = torch.sum(L*(torch.sum(1 - positive_indices*input + \
                                          negative_indices*input, dim=1, keepdim=True)),dim=1)
        #ctx.save_for_backward(input, target)
        ctx.L = L
        ctx.positive_indices = positive_indices
        ctx.negative_indices = negative_indices
        
        return torch.sum(loss, dim=0)

    # This function has only a single output, so it gets only one gradient 
    @staticmethod
    def backward(ctx, grad_output):
        #input, target = ctx.saved_variables
        L = Variable(ctx.L, requires_grad = False)
        positive_indices = Variable(ctx.positive_indices, requires_grad = False) 
        negative_indices = Variable(ctx.negative_indices, requires_grad = False)

        pos_grad = torch.sum(L,dim=1,keepdim=True)*(-positive_indices)
        neg_grad = torch.sum(L,dim=1,keepdim=True)*negative_indices
        grad_input = grad_output*(pos_grad+neg_grad)

        return grad_input, None, None

#--- main class
class WARPLoss(nn.Module): 
    def __init__(self, max_num_trials = None): 
        super().__init__()
        self.max_num_trials = max_num_trials
        
    def forward(self, input, target): 
        return WARP.apply(input.cpu(), target.cpu(), self.max_num_trials)

## WARP loss
#-- Weighted Approximate-Rank Pairwise loss
def loss_warp(outputs, labels):
    """
    Sigmoid + WARP loss
    """
    return WARPLoss()(torch.sigmoid(outputs),labels)



#############################
# Log-Sum-Exp-Pairwise Loss 
# - Reference: https://arxiv.org/pdf/1704.03135.pdf

def _to_one_hot(y, n_dims=None):
    """ 
    Take integer y (tensor or variable) with n dims and 
    convert it to 1-hot representation with n+1 dims
    """
    y_tensor = y.data if isinstance(y, Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(y.size()[0], -1)
    
    return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot

class LSEP(Function): 
    """
    Autograd function of LSEP loss. Appropirate for multi-label
    - Reference: Li+2017
      https://arxiv.org/pdf/1704.03135.pdf
    """
    
    @staticmethod
    def forward(ctx, input, target, max_num_trials = None):
        batch_size = target.size()[0]
        label_size = target.size()[1]

        ## rank weight 
        rank_weights = [1.0/1]
        for i in range(1, label_size):
            rank_weights.append(rank_weights[i-1] + (1.0/i+1))

        if max_num_trials is None: 
            max_num_trials = target.size()[1] - 1

        ##
        positive_indices = target.gt(0).float()
        negative_indices = target.eq(0).float()
        
        ## summing over all negatives and positives
        loss = 0.
        for i in range(input.size()[0]): # loop over examples
            pos = np.array([j for j,pos in enumerate(positive_indices[i]) if pos != 0])
            neg = np.array([j for j,neg in enumerate(negative_indices[i]) if neg != 0])
            
            for j,pj in enumerate(pos):
                for k,nj in enumerate(neg):
                    loss += np.exp(input[i,nj]-input[i,pj])
        
        loss = torch.from_numpy(np.array([np.log(1 + loss)])).float()
        
        ctx.save_for_backward(input, target)
        ctx.loss = loss
        ctx.positive_indices = positive_indices
        ctx.negative_indices = negative_indices
        
        return loss

    # This function has only a single output, so it gets only one gradient 
    @staticmethod
    def backward(ctx, grad_output):
        input, target = ctx.saved_variables
        loss = Variable(ctx.loss, requires_grad = False)
        positive_indices = ctx.positive_indices
        negative_indices = ctx.negative_indices

        fac  = -1 / loss
        grad_input = torch.zeros(input.size())
        
        ## make one-hot vectors
        one_hot_pos, one_hot_neg = [],[]
        
        for i in range(grad_input.size()[0]): # loop over examples
            pos_ind = np.array([j for j,pos in enumerate(positive_indices[i]) if pos != 0])
            neg_ind = np.array([j for j,neg in enumerate(negative_indices[i]) if neg != 0])
            
            one_hot_pos.append(_to_one_hot(torch.from_numpy(pos_ind),input.size()[1]))
            one_hot_neg.append(_to_one_hot(torch.from_numpy(neg_ind),input.size()[1]))
            
        ## grad
        for i in range(grad_input.size()[0]):
            for dum_j,phot in enumerate(one_hot_pos[i]):
                for dum_k,nhot in enumerate(one_hot_neg[i]):
                    grad_input[i] += (phot-nhot)*torch.exp(-input[i].data*(phot-nhot))
        ## 
        grad_input = Variable(grad_input) * (grad_output * fac)

        return grad_input, None, None
    
#--- main class
class LSEPLoss(nn.Module): 
    def __init__(self): 
        super(LSEPLoss, self).__init__()
        
    def forward(self, input, target): 
        return LSEP.apply(input.cpu(), target.cpu())

def loss_lsep(outputs, labels):
    """
    Sigmoid + LSEP loss
    """
    # print('LSEP')
    return LSEPLoss()(torch.sigmoid(outputs),labels)

