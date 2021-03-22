import torch
import torch.nn as nn
import copy


class MLP(nn.Module):
    """
    Implementation of A fully connected feed forward neural network
    """
    def __init__(self, input_size, output_size=2, layers = [2700, 500], bias=True, dropout=0.8):
        """
        Args:
            - input_size: dimension of the problem
            - output_size: 2 (for binary classification)
            - layers : hidden unit sizes
            - bias : If True, add a bias in the linear layers
        """
        super(MLP, self).__init__()
        self.layers = layers
        self.output_size = output_size
        if len(layers)>0:
            self.fc = [nn.Linear(input_size, self.layers[0],bias=bias)]
            for i in range(len(self.layers)-1):
                self.fc.append(nn.Linear(self.layers[i], self.layers[i+1],bias=bias))
                self.fc.append(nn.BatchNorm1d(self.layers[i+1]))
                self.fc.append(nn.ReLU(inplace=True))
                self.fc.append(nn.Dropout(dropout))
            self.fc.append(nn.Linear(self.layers[-1], output_size,bias=bias))
        else:
            self.fc = [nn.Linear(input_size, output_size,bias=bias)]
        self.linear = nn.Sequential(*self.fc)
    
    def forward(self, x):
        return self.linear(x)
    
def fine_tune_mlp(pretrained, freeze = False):
    """Creates a model from a pre-trained MLP model using Transfer Learning"""
    """
    Args:
        - pretrained : (nn.Module) pre-trained MLP model to be used.
        - freeze : (bool) if True, params in hidden layers will be freezed.
    """
    model = copy.deepcopy(pretrained)
    if freeze:
        for i in range(len(model.fc)-1) :
            for param in model.fc[i].parameters():
                param.requires_grad = False
    model.fc[len(model.fc)-1] = nn.Linear(model.fc[len(model.fc)-5].out_features, 2,bias=True)
    return model