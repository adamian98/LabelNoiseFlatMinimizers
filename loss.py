import torch
from torch import nn
from math import log
from torch.nn.functional import nll_loss

class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, p=0, label_noise=False,dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.p = p
        self.label_noise = label_noise
        self.C = num_classes
        self.dim = dim
        self.minloss = -(1-p)*log(1-p) - p*log(p/(num_classes-1)) if p > 0 else 0

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        if self.p==0:
            loss = nll_loss(pred,target)
            return loss,loss
        else:
            with torch.no_grad():
                smooth_dist = torch.zeros_like(pred)
                smooth_dist.fill_(self.p / (self.C - 1))
                smooth_dist.scatter_(1, target.data.unsqueeze(1), 1-self.p)
                if self.label_noise:
                    noisy_target = torch.multinomial(smooth_dist,1).squeeze()
            smooth_loss = torch.mean(torch.sum(-smooth_dist * pred, dim=self.dim))
            smooth_loss = smooth_loss - self.minloss
            if self.label_noise:
                return nll_loss(pred,noisy_target), smooth_loss
            else:
                return smooth_loss, smooth_loss