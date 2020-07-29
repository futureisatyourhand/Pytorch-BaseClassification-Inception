import torch 
import torch.nn as nn
class InceptionModule(nn.Module):
    def __init__(self,channels_in,channels_out):
        super(InceptionModule,self).__init__()
        self.inception1_0=nn.Conv2d(channels_in,channels_out[0],1,1,bias=True)
        self.inception1_1=nn.Sequential(
            nn.Conv2d(channels_in,channels_out[1],1,1,bias=True),
            nn.Conv2d(channels_out[1],channels_out[2],3,1,1,bias=True)
        )
        self.inception1_2=nn.Sequential(
            nn.Conv2d(channels_in,channels_out[3],1,1,bias=True),
            nn.Conv2d(channels_out[3],channels_out[4],5,1,2,bias=True),
        )
        self.inception1_3=nn.Sequential(
            nn.MaxPool2d(3,1,1),
            nn.Conv2d(channels_in,channels_out[5],1,1,bias=True),
        )
    def forward(self,x):
        x1=self.inception1_0(x)
        x2=self.inception1_1(x)
        x3=self.inception1_2(x)
        x4=self.inception1_3(x)
        return torch.cat([x1,x2,x3,x4],1)
class InceptionV1(nn.Module):
    def __init__(self,num_classes,flag=True):
        super(InceptionV1,self).__init__()

        self.num_classes=num_classes
        
        self.network=nn.Sequential(
            nn.Conv2d(3,64,7,2,bias=True),
            nn.MaxPool2d(3,2),

            nn.BatchNorm2d(64),
            nn.Conv2d(64,192,1,1,1,bias=True),
            nn.Conv2d(192,192,3,1,bias=True),
            nn.BatchNorm2d(192),
            nn.MaxPool2d(3,2),
        )
        self.inception1=InceptionModule(192,[64,96,128,16,32,32])
        ##concat:256
        self.inception2=InceptionModule(256,[128,128,192,32,96,64])
        self.max1=nn.MaxPool2d(3,2)
        ##concat:480
        self.inception3=InceptionModule(480,[192,96,208,16,48,64])
        ##concat:512
        self.inception4=InceptionModule(512,[160,112,224,24,64,64])
        ##concat:648
        self.inception5=InceptionModule(512,[128,128,256,24,64,64])
        ##concat:512
        self.inception6=InceptionModule(512,[112,144,288,32,64,64])
        ##concat:528
        self.inception7=InceptionModule(528,[256,160,320,32,128,128])
        ##concat:832
        self.max2=nn.MaxPool2d(3,2)
        self.inception8=InceptionModule(832,[256,160,320,32,128,128])
        ##concat:832
        self.inception9=InceptionModule(832,[384,192,384,48,128,128])
        self.avg=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(1024,num_classes),
        )
        
        self.softmax=nn.Softmax(-1)
        self.loss=nn.CrossEntropyLoss(reduce=True)
        self.flag=flag
    def forward(self,inputs,targets=None):
        inputs=self.network(inputs)
        inputs=self.inception1(inputs)
        inputs=self.inception2(inputs)
        inputs=self.max1(inputs)
        inputs=self.inception3(inputs)
        inputs=self.inception4(inputs)
        inputs=self.inception5(inputs)
        inputs=self.inception6(inputs)
        inputs=self.inception7(inputs)
        inputs=self.max2(inputs)
        inputs=self.inception8(inputs)
        inputs=self.inception9(inputs)
        inputs=self.avg(inputs)
        inputs=inputs.view(inputs.shape[0],-1)
        inputs=self.fc(inputs)
        if targets is not None:
            return self.loss(inputs,targets)
        else:
            return self.softmax(inputs)
