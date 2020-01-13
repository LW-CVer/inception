# _*_ coding:utf-8 _*_
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import  DataLoader
import model
from model import InceptionNet as model
from mydataset import Mydataset
#数据读取
transform=transforms.Compose([transforms.ToTensor()])#调整图片大小,torchvision版本较低，没有Resize方法

traindata=Mydataset("/app/datas/6/mini-imagenet/images/",data_type="train",transform=transform)

valdata=Mydataset("/app/datas/6/mini-imagenet/images/",data_type="val",transform=transform)

trainloader=DataLoader(traindata,batch_size=4,shuffle=True,num_workers=2)

valloader=DataLoader(valdata,batch_size=4,shuffle=True,num_workers=2)



#网络初始化
'''def weights_init(m):
    if isinstance(m,nn.Conv2d):
        m.weight.data.normal_(0.0,0.02)
        #xavier(m.weight.data)
    elif isinstance(m,nn.BatchNorm2d):
        m.weight.data.normal_(0.0,0.02)
        m.bias.data.fill_(0)
        #xavier(m.weight.data)
        #xavier(m.bias.data)'''
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
class_num=100 
      
net=model(class_num)

net.apply(weights_init)
#预加载模型
#net.load_state_dict(torch.load("...."))
criterion=nn.CrossEntropyLoss()

#optimizer=optim.SGD(net.parameters(),lr=0.0001,momentum=0.9)

optimizer = optim.Adam(net.parameters(), lr=0.001,
                           betas=(0.5, 0.999))
temp=1000

for epoch in range(0,20):
    val_accuracy=0
    val_all=0
    all_loss=0
    
    j=0
    for i,data in enumerate(trainloader,0):
        img,label=data
        img=Variable(img)
        label=Variable(label)
        #训练时才计算梯度
        for p in net.parameters():
            p.requires_grad=True
        net.train()
        optimizer.zero_grad()#每次反响传播前将梯度置0
        output=net(img)
        
        loss=criterion(output,label)
        loss.backward()
      
        if j%10==0:
            print("loss------------->",float(loss))
        j=j+1
        all_loss=all_loss+float(loss)#解决叠加loss张量导致的内存溢出问题
        optimizer.step()
    print("the Avg_loss is----------------------------> ",float(all_loss/j))
    print("--------start val--------")
    for i,data in enumerate(valloader,0):
        img,label=data
        val_all+=label.size(0)
        img=Variable(img)
        for p in net.parameters():
            p.requires_grad=False
        net.eval()
        output=net(img)
        _,predicted=torch.max(output.data,1)
        val_accuracy+=(label==predicted).sum()
    print("the val accuracy is:",float(val_accuracy/val_all))
        
    
    
    if temp>(all_loss/j) :
        print("saving model")
        model_name="./model/"+str(epoch)+"_inception.pth"
        torch.save(net.state_dict(),model_name)
        temp=all_loss/j
    
    
    
    
    
    