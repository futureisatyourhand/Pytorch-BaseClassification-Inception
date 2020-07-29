#---coding:utf-8--
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset
import os
from PIL import Image
class MyData(Dataset):
    def __init__(self,dirs,transforms=None):
        super(MyData,self).__init__()
        ##读取文件夹目录
        classes_dir=os.listdir(dirs)
        images=[]
        for classes in classes_dir:
            print(dirs,classes)
            if os.path.isdir(dirs+str(classes))==False:
                continue
            for f in os.listdir(dirs+str(classes)):
                if f.endswith(('png','jpg','jpeg','gif','bmp')):
                    images.append(dirs+str(classes)+"/"+f)
        self.images=images
        self.transforms=transforms
    def __getitem__(self,item):
        data=self.images[item]
        label=int(data.split('/')[-2])
        image=Image.open(data).convert('RGB')
        if self.transforms is not None:
            image=self.transforms(image)
        return image,label

    def __len__(self):
        return len(self.images)

        
