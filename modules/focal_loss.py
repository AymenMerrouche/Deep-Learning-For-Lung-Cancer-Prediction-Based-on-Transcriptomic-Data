import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.modules.loss._WeightedLoss):
    """
    Implementation of the focal loss for binary classification.
    """
    def __init__(self, weight=None, gamma=2,reduction='mean'):
        """
        :param weight: alpha in the focal loss formula. It balances the importance of negative and positive examples.
        :param gamma: focusing parameter, gamma in the focal loss formula. It down-weights easy examples. The bigger this parameter
        the faster the rate at which easy examples are downlighted is.
        """
        super(FocalLoss, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):

        ce_loss = F.cross_entropy(input, target,reduction=self.reduction,weight=self.weight) 
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss