import torch
import torch.nn as nn
import math

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class TwoLayerReluASI(nn.Module):

    def __init__(self, input_dim=1, num_neurons=100, initialization="default", bias_tune_tuple = ("default"), adaptive=False):
        super(TwoLayerReluASI, self).__init__()

        self.features1 = nn.Sequential(
            nn.Linear(input_dim, num_neurons),
            nn.ReLU(inplace=True),
            nn.Linear(num_neurons,1)
        )

        self.features2 = nn.Sequential(
            nn.Linear(input_dim, num_neurons),
            nn.ReLU(inplace=True),
            nn.Linear(num_neurons,1)
        )
        
        bias_tune = bias_tune_tuple[0]
        
        if bias_tune == "uniform":
            self.features1[0].bias.data.uniform_(-1*bias_tune_tuple[1], 1*bias_tune_tuple[1])
            
        if bias_tune == "normal":
            self.features1[0].bias.data.normal_(mean=0, std=1*bias_tune_tuple[1])
            print("bias_tune normal success")

        if bias_tune == "normal [-2,2]":
            self.features1[0].bias.data.normal_(mean=0, std=1*2)

        if bias_tune == "normal [-1,1]":
            self.features1[0].bias.data.normal_(mean=0, std=1)

        if bias_tune=="normal 0.1":
            self.features1[0].bias.data.normal_(mean=0, std=math.sqrt(0.1))

        if initialization=="+1-1":
            self.features1[0].weight.data = self.features1[0].weight.data.sign()

        if initialization=="normal":
            self.features1[0].weight.data.normal_(mean=0, std=1)
            
        if initialization=="unit_vector":
            self.features1[0].weight.data = self.features1[0].weight.data/torch.norm(self.features1[0].weight.data, dim=1).reshape(-1,1)

        if adaptive:
            self.features1[2].weight.data = self.features1[2].weight.data * (num_neurons**0.5)
            self.features1[2].bias.data = self.features1[2].bias.data * (num_neurons**0.5)
        
        self.features2[0].weight.data = self.features1[0].weight.data.clone()
        self.features2[0].bias.data = self.features1[0].bias.data.clone()
        self.features2[2].weight.data = self.features1[2].weight.data.clone()
        self.features2[2].bias.data = self.features1[2].bias.data.clone()
        
        self.adaptive = adaptive
        self.num_neurons = num_neurons
        

    def forward(self, x):
        if self.adaptive:
            x = ((math.sqrt(2)/2*self.features1(x)-math.sqrt(2)/2*self.features2(x)))/(self.num_neurons**0.5)
        else:
            x = math.sqrt(2)/2*self.features1(x)-math.sqrt(2)/2*self.features2(x)
        return x

class TwoLayerRelu(nn.Module):

    def __init__(self, input_dim=1, num_neurons=100, initialization="default", bias_tune_tuple = "default", spiky = False):
        super(TwoLayerRelu, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(input_dim, num_neurons),
            nn.ReLU(inplace=True),
            nn.Linear(num_neurons,1)
        )
        bias_tune = bias_tune_tuple[0]
        
        if bias_tune == "uniform":
            self.features[0].bias.data.uniform_(-1*bias_tune_tuple[1], 1*bias_tune_tuple[1])
        
        if bias_tune == "normal":
            self.features[0].bias.data.normal_(-1*bias_tune_tuple[1], 1*bias_tune_tuple[1])
#             print("bias_tune normal success")
        
        if bias_tune == "normal [-2,2]":
            self.features[0].bias.data.normal_(mean=0, std=1*2)

        if bias_tune == "normal [-1,1]":
            self.features[0].bias.data.normal_(mean=0, std=1)

        if bias_tune == "normal 0.1":
            self.features[0].bias.data.normal_(mean=0, std=math.sqrt(0.1))

        if initialization == "+1-1":
            self.features[0].weight.data = self.features[0].weight.data.sign()

        if initialization == "normal":
            self.features[0].weight.data.normal_(mean=0, std=1)

        if spiky:
            self.features[0].bias.data = torch.tensor()
        # for idx, p in enumerate(self.features):
        #     if p.__class__.__name__=="Linear":
        #         if idx==0:
        #             stdv = math.sqrt(1. / math.sqrt(p.weight.size(0)))
        #         if idx==2:
        #             stdv = math.sqrt(1. / math.sqrt(p.weight.size(1)))*initialization
        #         p.weight.data.uniform_(-stdv, stdv)
        #         if p.bias is not None:
        #             if idx==0 and bias_tune:
        #                 p.bias.data.uniform_(-stdv*2, stdv*2)
        #             else:
        #                 p.bias.data.uniform_(-stdvs, stdv)

    def forward(self, x):
        x = self.features(x)
        return x

class AlexNet(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.normalize_layer=[1, 5, 9, 12, 15]
        self.weight_layer=[0, 1, 4, 5, 8, 9, 11, 12, 14, 15]
        self.scalable_weight_layer = list(set(self.weight_layer)-set(self.normalize_layer))

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class ShallowNet(nn.Module):

    def __init__(self, num_classes=10):
        super(ShallowNet, self).__init__()

        self.normalize_layer=[1, 5, 10 ,14]
        self.weight_layer=[0, 1, 4, 5, 9, 10, 13, 14]
        self.scalable_weight_layer = list(set(self.weight_layer)-set(self.normalize_layer))

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            Flatten(),
            nn.Linear(3136, 384),
            nn.BatchNorm1d(384),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(384, 192),
            nn.BatchNorm1d(192),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.classifier = nn.Linear(192, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class DeepNet(nn.Module):

    def __init__(self, num_classes=10):
        super(DeepNet, self).__init__()

        self.normalize_layer=[1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 55]
        self.weight_layer=[0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29, 32, 33, 36, 37, 40, 41, 44, 45, 48, 49, 54, 55]
        self.scalable_weight_layer = list(set(self.weight_layer)-set(self.normalize_layer))

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True), nn.Dropout2d(p=0.3),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True), nn.Dropout2d(p=0.4),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True), nn.Dropout2d(p=0.4),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True), nn.Dropout2d(p=0.4),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True), nn.Dropout2d(p=0.4),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True), nn.Dropout2d(p=0.4),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True), nn.Dropout2d(p=0.4),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True), nn.Dropout2d(p=0.4),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True), nn.Dropout(p=0.5),
        )
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
