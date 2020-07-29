import torch.nn as nn
class Classification(nn.Module):
    #input size:128*128;
    def __init__(self,num_classes,flag=True):
        super(Classification,self).__init__()
        self.num_classes=num_classes
        self.flag=flag
        self.convs=nn.Sequential(
            nn.Conv2d(3,64,3,1,1,bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            #case1:nn.AvgPool2d(3,2),
            #case2:nn.MaxPool2d(3,2),

            nn.Conv2d(64,128,3,1,1,bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            #case1 and case2:nn.MaxPool2d(3,2),

            nn.Conv2d(128,256,3,2,1,bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256,512,3,1,1,bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,2),#case3

            nn.Conv2d(512,512,3,1,1,bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,2),#case3

            nn.Conv2d(512,728,3,2,1,bias=True),
            nn.BatchNorm2d(728),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1)),#case4
        )
        self.fc=nn.Sequential(
            #case1 and case2 and case3:nn.Linear(11648,1024),
            #case1 and case2 and case3:nn.ReLU(inplace=True),
            #case1 and case2 and case3:nn.Dropout(0.5),
            #case1 and case2 and case3:nn.Linear(1024,2),
            nn.Linear(728,2),#case4
           
        )
        self.classes= nn.Softmax(dim=-1)
        self.loss=nn.CrossEntropyLoss(reduce=True)
    def forward(self,x,targets):
        x=self.convs(x)
        x=x.view(x.shape[0],-1)
        x=self.fc(x)
        if self.flag:
            loss=self.loss(x,targets)
            return loss
        else:
            return self.classes(x)
