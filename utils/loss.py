import torch, pdb
import torch.nn.functional as F
import torch.nn as nn
import math
class SELoss(nn.Module):
    '''
            Semantic Encoding Loss：像素级的交叉熵损失函数无法考虑到全局信息，可能会导致小目标无法正常识别，
            而SELoss可以平等的考虑不同大小的目标。SELoss损失的target是一个（N, NUM_CLASSES）的矩阵，
            它的构造也很简单，如果图片中存在某种物体，则对应的target的标签就为1
            '''
    def __init__(self):
        super(SELoss, self).__init__()
        #self.mse_loss = nn.MSELoss(reduction='sum')
        self.loss = nn.L1Loss(reduction='sum')
    def forward(self, predict, groundth):

        pos_inds = groundth.ne(255).float()
        #.ne()逐元素比较,return为二值图
        predict = predict * pos_inds
        groundth = groundth * pos_inds
        
        return self.loss(predict, groundth)

        
class OHEMSELoss(nn.Module):
    def __init__(self):
        super(OHEMSELoss, self).__init__()
        self.rate = 0.5
    def forward(self, predict, groundth, keep_num):
    
        pos_inds = groundth.gt(0).float()
        predict = predict * pos_inds
        
        ohem_reg_loss = torch.abs(predict - groundth).view(-1)
        
        sorted_ohem_loss, idx = torch.sort(ohem_reg_loss, descending=True)
        
        keep_num = int(self.rate * keep_num)
        
        keep_idx = idx[:keep_num]
        ohem_reg_loss = ohem_reg_loss[keep_idx]
        
        return ohem_reg_loss.sum() / keep_num

class weightedMSE(nn.Module):
    def forward(self,pre,target):
        #target 0-1
        # pre = torch.sigmoid(pre)
        # print(torch.max(pre), torch.min(pre))
        # b
        loss = torch.pow((pre-target),2)
        # loss = torch.abs(pre-target)

        #weight_mask = (target+0.1)/1.1
        weight_mask = target*5+1
        # weight_mask = torch.pow(target,2)*8+1

        #gamma from focal loss
        #gamma = torch.pow(torch.abs(target-pre), 2)

        loss = loss*weight_mask#*gamma

        loss = torch.sum(loss)/target.shape[0]/target.shape[1]

        # bg_loss = self.bgLoss(pre, target)
        return loss
        
         


class FocalLoss(nn.Module):
    def forward(self, pred, gt, pos_weights, keep_mask=None):
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        neg_weights = torch.pow(1 - gt, 4)
        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_weights
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

        if keep_mask is not None:
            pos_loss = (pos_loss * keep_mask).sum()
            neg_loss = (neg_loss * keep_mask).sum()
        else:
            pos_loss = pos_loss.sum()
            neg_loss = neg_loss.sum()
        return -(pos_loss + neg_loss)



class Swingloss(nn.Module):
    def __init__(self):
        super(Swingloss, self).__init__()
        self.w1 = 1.5
        self.w2=10
        self.epsilon = 2
        self.constant =self.w1 - self.w2 * math.log(1 + self.w1/ self.epsilon)
        # print(self.w, self.epsilon, self.constant)
        self.weight=8
    def forward(self, prediction, gt):
        weight_mask =gt * self.weight + 1
        diff = torch.pow((prediction-gt),2)
        loss = torch.where(diff < self.w1,diff,self.w2 * torch.log(1 + diff / self.epsilon)+self.constant)
        #print(torch.sum(loss))
        loss=loss*weight_mask
        loss = loss.view(-1)
        #print('loss',torch.sum(loss))
        return torch.sum(loss)
