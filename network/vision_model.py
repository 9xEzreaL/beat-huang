import torch
import torch.nn.functional as F
import torch.nn as nn
from network.model import ResNormLayer
from torchvision import models
import pretrainedmodels


_all = [
    'my_densenet',
    'resnext',
    'resnest'
]

class Flatten(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x.view(x.data.size(0),-1)


class hardnet(nn.Module):
    def __init__(self, num_classes, use_meta=False, pretrain=True):
        super(hardnet, self).__init__()
        self.use_meta = use_meta
        model = torch.hub.load('PingoLH/Pytorch-HarDNet', 'hardnet85', pretrained=pretrain)
        self.model_f = model.base[:19]
        self.base = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                Flatten(),
                nn.Dropout(0.2))
        self.base_f = nn.Sequential(nn.Linear(1280, num_classes, bias=True))


    def forward(self, input, meta=None):
        for layer in self.model_f:
            input = layer(input)
        output_feature = input
        # output_feature = self.model_f(input)
        output = self.base_f(output_feature)

        return output

class efficientnet(nn.Module):
    def __init__(self, num_classes, use_meta=False, pretrain=True):
        super(efficientnet, self).__init__()
        self.use_model = use_meta
        self.model = models.efficientnet_b4(pretrained=pretrain)
        print(self.model)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.meta = nn.Sequential(
            nn.Linear(1792, 896),
            nn.ReLU(inplace=True),
            nn.LayerNorm(896),
            ResNormLayer(896),
        )
        self.model.classifier[1] = nn.Linear(896, num_classes, bias=True)


    def forward(self, input, meta=None):
        output_feature = self.model.features(input)
        output_feature = self.avgpool(output_feature)
        output_feature = torch.flatten(output_feature, 1)
        output_feature = self.meta(output_feature)
        output = self.model.classifier(output_feature)
        # output = self.model(input)

        return output


class my_densenet(nn.Module):
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


class densenet_201(nn.Module):
    def __init__(self, num_classes, use_meta=False, pretrain=True):
        super(densenet_201, self).__init__()
        self.use_meta = use_meta
        self.model = models.densenet201(pretrained=pretrain)
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features=in_features, out_features=num_classes, bias=True)

    def forward(self, input, meta=None):
        output = self.model(input)
        return output

# not work
class arc_efficientnet(nn.Module):
    def __init__(self, num_classes, use_meta=False, pretrain=True):
        super(arc_efficientnet, self).__init__()
        self.use_model = use_meta
        self.model = models.efficientnet_b4(pretrained=pretrain)
        self.model.classifier[1] = nn.Linear(1792, num_classes, bias=True)
        self.class_weight = nn.Parameter(torch.rand(num_classes, 1792+384))
        self.logit_scale = nn.Parameter(torch.rand(()))

    def forward(self, input, meta=None, label=None):
        if meta is not None:
            meta = meta[:, :2]  # no time
            output_feature = self.model.features(input)
            output_feature = output_feature.mean(dim=(2, 3))
            output_feature = output_feature / output_feature.norm(dim=-1, keepdim=True)
            cw = self.class_weight
            cw = cw / cw.norm(dim=-1, keepdim=True)
            out_meta = self.meta(meta)
            agg_out = torch.cat([output_feature, out_meta], 1)
            logits = agg_out @ cw.t()
        else:
            output_feature = output_feature.mean(dim=(2, 3))
            output_feature = output_feature / output_feature.norm(dim=-1, keepdim=True)
            cw = self.class_weight
            cw = cw / cw.norm(dim=-1, keepdim=True)
            logits = output_feature @ cw.t()

        # if label is not None:
        #     # Do the computation for arcface
        #     cos_logits = (1 - logits * logits).clamp(1e-8, 1 - 1e-8) ** 0.5
        #     # one_hot = F.one_hot(label, 33)
        #     logits = torch.where(label == 1,
        #                          0.8 * logits - 0.6 * cos_logits,
        #                          logits)
        return logits * self.logit_scale.exp()


class Resnext(nn.Module):
    def __init__(self, num_classes, use_meta=False, pretrain=True):
        super(Resnext, self).__init__()
        self.use_meta = use_meta
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d', pretrained=pretrain)

        in_features = 2048
        # self.model.fc = nn.Linear(in_features=in_features, out_features=num_classes, bias=True)
        self.model.fc = nn.Sequential(  nn.Linear(in_features=in_features, out_features=1024, bias=True),
                                        nn.ReLU(),
                                        nn.Dropout(0.5),
                                        nn.Linear(in_features=1024, out_features=512, bias=True),
                                        nn.ReLU(),
                                        nn.Dropout(0.5),
                                        nn.Linear(in_features=512, out_features=num_classes, bias=True)
                                        )
        # print(self.model)


    def forward(self, input, meta=None):
        output = self.model(input)
        return output


class Resnest(nn.Module):
    def __init__(self, num_classes, use_meta=False, pretrain=True):
        super(Resnest, self).__init__()
        self.use_meta = use_meta
        torch.hub.list('zhanghang1989/ResNeSt', force_reload=True)
        self.model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=pretrain)
        in_features = 2048
        self.model.fc = nn.Sequential(  nn.Linear(in_features=in_features, out_features=1024, bias=True),
                                        nn.ReLU(),
                                        nn.Dropout(0.3),
                                        nn.Linear(in_features=1024, out_features=512, bias=True),
                                        nn.ReLU(),
                                        nn.Dropout(0.3),
                                        nn.Linear(in_features=512, out_features=num_classes, bias=True)
                                        )
        # print(self.model)

    def forward(self, input, meta=None):
        output = self.model(input)
        return output