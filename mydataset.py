import torch
import util
from util import train_type as key
from PIL import Image
class Mydataset(torch.utils.data.Dataset):
       
    def __init__(self,path,data_type,transform=None):
        super(Mydataset,self).__init__()
        self.imgs,self.labels=util.get_data(data_type)#只加载路径
        self.path=path
        self.transform=transform
        assert len(self.imgs)==len(self.labels)
    def __getitem__(self,index):#
        label=self.labels[index]
        img_path=self.path+self.imgs[index].strip()
        img=Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img=img.resize((299,299))#图像会变形
            img=self.transform(img)
        label=key.index(label)
        return img,label
        
    def __len__(self):
        return len(self.imgs)