import torch
import torch.nn.functional as F
import torch.nn as nn
from network.model import ResNormLayer
from torchvision import models
import pretrainedmodels

class BaseModel(nn.Module):
    def __init__(self, num_classes, use_meta=False, pretrain=False):
        super(my_densenet, self).__init__()
        self.model = models.densenet121(pretrained=pretrain)
        self.mp = nn.AdaptiveAvgPool2d((1, 1))
        in_features = self.model.classifier.in_features
        self.mid_layer = nn.Sequential(nn.Linear(in_features=in_features, out_features=512, bias=True),
                                        nn.ReLU(),
                                        nn.Dropout(0.5)
                                        )
        self.classifier = nn.Sequential(nn.Linear(in_features=512, out_features=num_classes, bias=True))


    def forward(self, input, meta=None):
        output_feature = self.model.features(input)
        output_feature = self.mp(output_feature)
        output_feature = torch.flatten(output_feature, 1)
        output_feature = self.mid_layer(output_feature)
        output = self.classifier(output_feature)
        return output
