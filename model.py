import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.nn import Parameter

class Vanilla_ResNet18(nn.Module):
    def __init__(self, args, num_classes=7):
        super(Vanilla_ResNet18, self).__init__()
        resnet18 = models.resnet18(pretrained=False)
        self.feature_extractor = nn.Sequential(*list(resnet18.children())[:-1]) 
        
        if('eye' in args.dataset):
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),  
                resnet18.bn1,
                resnet18.relu,
                resnet18.maxpool,
                resnet18.layer1,
                resnet18.layer2
            )
            
        self.layer3 = resnet18.layer3 
        self.layer4 = resnet18.layer4  
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        num_ftrs = resnet18.fc.in_features
        self.fc = nn.Linear(num_ftrs, num_classes)
        
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.fc(x)
        return logits
    
    def forward_feature(self, x):
        x = self.feature_extractor(x)
        x = self.avgpool(x)
        feature = torch.flatten(x, 1)
        logits = self.fc(feature)
        return logits, feature
    
    
class VirtualCenter(nn.Module):
    def __init__(self, num_ftrs, num_classes):
        super(VirtualCenter, self).__init__()
        self.center = Parameter(torch.Tensor(num_ftrs, num_classes))
        self.center.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

class fair_resnet18(nn.Module):
    def __init__(self, args, num_classes=7, sensitive_attributes=2,in_channel=3):
        super(fair_resnet18, self).__init__()
        self.args = args
        if('eye' in args.dataset):
            in_channel = 1
        self.num_classes = num_classes
        resnet18 = models.resnet18(pretrained=False)
        num_ftrs = resnet18.fc.in_features
        self.num_ftrs = num_ftrs
        self.sensitive_attributes = sensitive_attributes
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.feature_extractor_unique = nn.Sequential(*list(resnet18.children())[:-1])
        if(in_channel==1):
            self.feature_extractor_unique = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False), 
                resnet18.bn1,
                resnet18.relu,
                resnet18.maxpool,
                resnet18.layer1,
                resnet18.layer2,
                resnet18.layer3,
                resnet18.layer4,
            )
            
        self.virtual_centers = nn.ModuleDict()
        self.decoupled_classifiers = nn.ModuleDict()
        for i in range(sensitive_attributes):
            self.virtual_centers[str(i)] = VirtualCenter(num_ftrs, num_classes)
            self.decoupled_classifiers[str(i)] = nn.Linear(num_ftrs, num_classes)
        self.fair_attr_classifier = nn.Linear(num_ftrs, sensitive_attributes)
            
    def forward(self, inputs):  
        feature = self.feature_extractor_unique(inputs)
                
        feature = self.avgpool(feature)
        feature = torch.flatten(feature,1)
        
        outputs, virtual_centers = [], []
        for i in range(self.sensitive_attributes):
            outputs.append(self.decoupled_classifiers[str(i)](feature) )
            
            normalized_feature = F.normalize(feature, dim=1)
            virtual_centers.append(normalized_feature.mm(F.normalize(self.virtual_centers[str(i)].center, dim=0)))
        
        outputs = torch.stack(outputs).to(device=feature.device) 
        virtual_centers = torch.stack(virtual_centers).to(device=feature.device) 
        
        attr_prediction = self.fair_attr_classifier(feature)    
        
        
        return outputs, virtual_centers, feature, attr_prediction
    
class Fair_Identity_Normalizer(nn.Module):
    def __init__(self, num_attr=0, dim=0, mu=0.001, sigma=0.1, momentum=0, test=False):
        super().__init__()
        self.num_attr = num_attr
        self.dim = dim

        self.mus = nn.Parameter(torch.randn(self.num_attr, self.dim)*mu)
        self.sigmas = nn.Parameter(torch.randn(self.num_attr, self.dim)*sigma)
        if test:
            self.sigmas = nn.Parameter(torch.ones(self.num_attr, self.dim)*sigma)
        self.eps = 1e-6
        self.momentum = momentum


    def forward(self, x, attr):
        x_clone = x.clone()
        for idx in range(x.shape[0]):
            x[idx,:] = (x[idx,:] - self.mus[attr[idx], :])/( torch.log(1+torch.exp(self.sigmas[attr[idx], :])) + self.eps)
        x = (1-self.momentum)*x + self.momentum*x_clone

        return x

    def __repr__(self):
        if self.mus is not None and self.sigmas is not None:
            sigma = torch.log(1+torch.exp(self.sigmas))
            sigma = torch.mean(sigma, dim=1)
            mu = torch.mean(self.mus, dim=1)
            out_str = ', '.join([f'G{i}: ({mu[i].item():f}, {sigma[i].item():f})' for i in range(mu.shape[0])])
        else:
            out_str = 'Attribute-Grouped Normalizer is not initialized yet.'
        return out_str
 
class FIN_ResNet18(nn.Module):
    def __init__(self, args, num_classes=7, sensitive_attributes=2, in_channel=3):
        super(FIN_ResNet18, self).__init__()
        resnet18 = models.resnet18(pretrained=False)
        self.feature_extractor = nn.Sequential(*list(resnet18.children())[:-4]) 
        if(in_channel==1):
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),  
                resnet18.bn1,
                resnet18.relu,
                resnet18.maxpool,
                resnet18.layer1,
                resnet18.layer2
            )
            
        self.layer3 = resnet18.layer3 
        self.layer4 = resnet18.layer4  
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        num_ftrs = resnet18.fc.in_features
        self.fc = nn.Linear(num_ftrs, num_classes)
        self.ag_norm = Fair_Identity_Normalizer(sensitive_attributes, dim=num_ftrs, mu=args.args, sigma=args.sigma, momentum=args.momentum)
    def forward(self, x, sex):
        x = self.feature_extractor(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        nml_feat =self.ag_norm(x, sex)
        logits = self.fc(nml_feat)
        return logits
    