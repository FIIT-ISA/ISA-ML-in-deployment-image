import torch
import torch.nn as nn
import timm

import torchvision.transforms as TVF

class MLP(nn.Module):
    def __init__(self, nin, nhidden, nout, dropout_p=0.5):
        super().__init__()
        if (nhidden == 0):
            self.main = nn.Linear(nin, nout)
        else:
            self.main = nn.Sequential(
                nn.Linear(nin, nhidden),
                nn.BatchNorm1d(nhidden),
                nn.ReLU(),
                nn.Dropout(dropout_p),
                nn.Linear(nhidden, 256),  
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(dropout_p),
                nn.Linear(256, nout)
            )

    def forward(self, x):
        # Our model now returns logits!
        logits = self.main(x)
        return logits
    


class SimpleConvModel(nn.Module):
    def __init__(
        self,
        chin,
        channels,
        num_hidden,
        num_classes
    ):
        super().__init__()
        self.num_classes = num_classes

        def conv(chin, chout, k, s, p):
            return nn.Sequential(
                nn.Conv2d(chin, chout, kernel_size=k, stride=s, padding=p),
                nn.BatchNorm2d(chout),
                nn.ReLU()
            )
        
        channels = [chin, 32, 64, 128, 256, 512]

        self.feature_extractor = nn.Sequential(
            conv(3, 16, 5, 1, 1),   # 256 -> 128
            nn.MaxPool2d(2,2),
            conv(16, 32, 3, 1, 1),   # 128 -> 64
            nn.MaxPool2d(2,2),
            conv(32, 64, 3, 1, 1),   # 64 -> 32
            nn.MaxPool2d(2,2),
            conv(64, 128, 3, 1, 1),   # 32 -> 16
            nn.MaxPool2d(2,2),
            conv(128, 256, 3, 1, 1),   # 16 -> 8
            nn.MaxPool2d(2,2),
            conv(256, 512, 3, 1, 1),   # 16 -> 8
            nn.MaxPool2d(2,2),
        )

        self.num_features = 512

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),                # 8x8 -> 1x1
            nn.Flatten(),                               # <B,C,1,1> -> <B,C>
            MLP(
                nin=1*1*self.num_features,
                nhidden=num_hidden*2,
                nout=num_classes
            )
        )


    def forward(self, x):
        f = self.feature_extractor(x)
        logits = self.head(f)
        #logits = logits.view(logits.size(0), -1)
        return logits
    

class PretrainedConvModel(nn.Module):
    def __init__(
        self,
        num_hidden,
        num_classes
    ):
        super().__init__()
        self.num_classes = num_classes

        # Create our pre-trained model
        self.feature_extractor = timm.create_model("resnet18", pretrained=True)
        self.feature_extractor.reset_classifier(0, "")

        # Freeze parameters
        for p in self.feature_extractor.parameters():
            p.requires_grad = False

        self.norm_transform = TVF.Normalize(
            mean=self.feature_extractor.default_cfg["mean"],
            std=self.feature_extractor.default_cfg["std"]
        )

        # Replace our own classification head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),                # 7x7 -> 1x1
            nn.Flatten(),                               # <B,C,1,1> -> <B,C>
            MLP(
                nin=1*1*self.feature_extractor.num_features,
                nhidden=num_hidden,
                nout=num_classes
            )
        )


    def forward(self, x):
        x = self.norm_transform(x)
        f = self.feature_extractor(x)
        logits = self.head(f)
        return logits
    
