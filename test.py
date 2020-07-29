from utils import collate_fn
from data import MyData
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
from argparse import ArgumentParser
from torch.backends import cudnn
from torch.autograd import Variable
import os
import numpy as np
from model import Classification
parse=ArgumentParser()
parse.add_argument('--models',type=str,default='./models/',help='')
parse.add_argument('--batch',type=int,default=32,help='')
parse.add_argument('--data',type=str,default='./test_set/test_set/',help='')
parse.add_argument('--no_cuda',action='store_true',default=False,help='')
args=parse.parse_args()
args.cuda=not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(1)
if args.cuda:
    torch.cuda.manual_seed(1)
    cudnn.benchmark=True

transform=transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
        ])
dataset=MyData(args.data,transform)
dataloader=DataLoader(dataset,batch_size=args.batch,shuffle=False,num_workers=4,collate_fn=collate_fn)
for f in os.listdir(args.models):
    model=Classification(2,False)
    model.load_state_dict(torch.load(args.models+f)['models'])
    model.eval()
    if args.cuda:
        model.cuda()
    cnt=0.0
    for i,(image,label) in enumerate(dataloader):
        image=Variable(image.cuda(),requires_grad=False)
        output=model(image,None)  
        output=output.cpu().detach().numpy()
        index=np.argmax(output,1)
        cnt+=sum(np.array(index,dtype=np.int)==label.detach().numpy())
    print(f,"acc:{:.4f}".format(cnt/len(dataset)))

