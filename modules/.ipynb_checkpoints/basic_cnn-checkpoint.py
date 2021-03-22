import torch
import torch.nn as nn
import copy

class Basic_CNN(nn.Module):  
    """
    A convolutional neural network with two convolutional layers
    """
    def __init__(self):
        super(Basic_CNN, self).__init__()

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(1, 12, kernel_size=8, stride=1, padding=0),
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.8),
            # Defining another 2D convolution layer
            nn.Conv2d(12, 12, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2)
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(12 * 40 * 40, 160),
            nn.BatchNorm1d(160),
            nn.ReLU(inplace=True),
            nn.Dropout(0.8),
             nn.Linear(160, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),
        )
           
        
        self.classify = nn.Linear(100, 2)

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return self.classify(x) #torch.sigmoid
    
    
def fine_tune_CNN(pretrained, freeze = False):
    """Creates a CNN model from a pre-trained model using Transfer Learning"""
    """
    Args:
        - pretrained : (nn.Module) pre-trained model to be used.
        - freeze : (bool) if True, params in convolutional layers will be freezed.
    """
    model = copy.deepcopy(pretrained)
    if freeze:
        for param in model.cnn_layers.parameters():
            param.requires_grad = False
    model.linear_layers = nn.Sequential(
            nn.Linear(12 * 40 * 40, 160),
            nn.BatchNorm1d(160),
            nn.ReLU(inplace=True),
            nn.Dropout(0.8),)
    model.classify = nn.Linear(160, 2)
    return model