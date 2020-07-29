#---coding:utf-8---
from argparse import ArgumentParser ############很重要的动态参数设置
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from data import MyData
from utils import *
from model import Classification
from torch.backends import cudnn
from torch.autograd import Variable
parse=ArgumentParser()
parse.add_argument("--datas",default="./training_set/training_set/",type=str,help="datas for training")
parse.add_argument("--test",default="./test_set/test_set/",type=str,help="datas for testing")
parse.add_argument("--epochs",default=100,type=int,help="epochs for training")
parse.add_argument("--batch",type=int,default=32,help="batch size for training")
parse.add_argument("--lr",type=float,default=0.00001,help="initial learning rate")
parse.add_argument("--beta1",type=float,default=0.99,help="beta1 for adam optimizer")
parse.add_argument("--beta2",type=float,default=0.999,help="beta2 for adam optimizer")
parse.add_argument("--seed",type=int,default=1,metavar="S",help="")
parse.add_argument("--save",type=str,default="models",help="dirs for saving models")
parse.add_argument("--log",type=int,default=20,help="")
parse.add_argument("--no_cuda",action="store_true",default=False,help="")
args=parse.parse_args()

#####数据处理
normalize=transforms.Compose([
            transforms.Resize((64,64)),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
            ])
##判断gpu是否可用并生成随机种子
args.cuda=not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
model=Classification(2)
if args.cuda:
    model.cuda()
    cudnn.benchmark=True

batch_size=args.batch
##构建数据迭代器  训练
dataset=MyData(args.datas,transforms=normalize)
valid=MyData(args.test,transforms=normalize)
dataloader=DataLoader(dataset,shuffle=True,batch_size=batch_size,num_workers=2,collate_fn=collate_fn)
valid_dataloader=DataLoader(valid,shuffle=True,batch_size=batch_size,num_workers=2,collate_fn=collate_fn)
##定义优化器
optimizer=torch.optim.Adam([{'params':model.parameters()}],lr=args.lr,betas=(args.beta1,args.beta2))
nums=len(dataloader)
logs=open('logs.txt','a+')
valid_logs=open('valid.txt','a+')
for epoch in range(args.epochs):
    for step,(images,labels) in enumerate(dataloader):
        images,labels=images.cuda(),labels.cuda()
        images,labels=Variable(images,requires_grad=False),Variable(labels,requires_grad=False)
        ##将梯度初始化为零（因为一个batch的loss关于weight的导数是所有sample的loss关于weight的导数的累加和）
        optimizer.zero_grad()##
        losses=model(images,labels)

        losses.backward()##
        ##更新所有参数
        optimizer.step()##
        print("epochs:{} step:{},losses:{:.4f},lr:{:.6f}".format(epoch+1,epoch*nums+step,losses.item(),optimizer.state_dict()['param_groups'][0]['lr']))
        logs.write(str(epoch*nums+step)+" "+str(losses.item())+" "+str(optimizer.state_dict()['param_groups'][0]['lr'])+"\n")
    if epoch%args.log==0:
        torch.save({'models':model.state_dict(),'optimzer':optimizer.state_dict(),'epochs':epoch},"./models/classes_{}.pth".format(epoch))
    if epoch%4==0:
        for s,(images,labels) in enumerate(valid_dataloader):
            images,labels=Variable(images.cuda(),requires_grad=False),Variable(labels.cuda(),requires_grad=False)
            losses=model(images,labels)
            valid_logs.write(str(epoch+1)+" "+str(losses.item())+"\n")
logs.close()
valid_logs.close()
