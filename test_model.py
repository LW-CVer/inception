#_*_ coding:utf-8 _*_
import torch
import model
from model import InceptionNet as model
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image

class_type=('plane','car','bird','cat','deer',
            'dog','frog','horse','ship','truck') 
num_class=10
net=model(num_class)
net.load_state_dict(torch.load("model3_inception.pth"))
net.eval()
img=Image.open("1.jpg").convert('RGB')
transform=transforms.Compose([transforms.Scale(299),transforms.ToTensor()])
img=transform(img)
img=img.unsqueeze(0)#增加batch维度
#print(img.shape)
img=Variable(img)
_,predicted=torch.max(net(img),1)
temp=predicted.data
index=int(temp)
print("图片中的是：",class_type[index])