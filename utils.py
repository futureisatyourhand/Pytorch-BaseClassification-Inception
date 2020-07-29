#---coding:utf-8--
import numpy as np
import torch

###数据加载时的每次batchs的方式
def collate_fn(images):
    image=[]
    labels=[]
    for i,data in enumerate(images):
        image.append(data[0])
        labels.append(data[1])
    return torch.stack(image,0),torch.from_numpy(np.array(labels,dtype='int'))
