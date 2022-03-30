import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
from torch.autograd import Variable, backward
import torch.backends.cudnn as cudnn


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

def weights_init_eye(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.eye_(m.weight)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

class Fc_cls(nn.Module):
    def __init__(self, num_class):
        super(Fc_cls, self).__init__()
        self.num_class = num_class
        self.classifier_fc = nn.Linear(1024, 1024, bias=False)
        self.classifier_norm_fc = nn.LayerNorm(1024, eps=1e-12)
        self.classifier_norm_fc.bias.requires_grad_(False)
        self.classifier_act_fc = nn.Tanh()
        self.classifier = nn.Linear(1024, num_class, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.classifier_fc.apply(weights_init_classifier)
    
    def forward(self, x):

        x = self.classifier_act_fc(self.classifier_norm_fc(self.classifier_fc(x)))
        x = self.classifier(x)

        return x

class Fc(nn.Module):
    def __init__(self, num_class):
        super(Fc, self).__init__()
        self.num_class = num_class
        self.classifier_fc = nn.Linear(1024, 1024, bias=False)
        self.classifier_norm_fc = nn.LayerNorm(1024, eps=1e-12)
        self.classifier_norm_fc.bias.requires_grad_(False)
        self.classifier_act_fc = nn.Tanh()

        self.classifier_fc.apply(weights_init_classifier)
    
    def forward(self, x):

        x = self.classifier_act_fc(self.classifier_norm_fc(self.classifier_fc(x)))

        return x