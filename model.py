#_*_ coding:utf-8 _*_
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
class InceptionNet(nn.Module):
    
    def __init__(self,num_classes):
        super(InceptionNet,self).__init__()
        
        self.pool=nn.MaxPool2d(3,2)
        self.conv1=nn.Conv2d(3,32,3,2,bias=False)#输入通道、输出通道、卷积核大小、步长
        self.bn1=nn.BatchNorm2d(32,eps=0.001)#输出通道数、防止标准化函数分母为0
        
        self.conv2=nn.Conv2d(32,32,3,1,bias=False)
        self.bn2=nn.BatchNorm2d(32,eps=0.001)
        
        self.conv3=nn.Conv2d(32,64,3,1,padding=1,bias=False)
        self.bn3=nn.BatchNorm2d(64,eps=0.001)
        
        
        self.conv4=nn.Conv2d(64,80,1,1,bias=False)
        self.bn4=nn.BatchNorm2d(80,eps=0.001)
        
        self.conv5=nn.Conv2d(80,192,3,1,bias=False)
        self.bn5=nn.BatchNorm2d(192,eps=0.001)
        
        self.Inception1_1=Inception1(192,32)
        self.Inception1_2=Inception1(256,64)
        self.Inception1_3=Inception1(288,64)
        
        self.Inception2_1=Inception2(288)
        self.Inception3_1=Inception3(768)
        self.Inception4_1=Inception4(768)
        self.Inception4_2=Inception4(768)
        self.Inception5_1=Inception5(768)
        self.Inception6_1=Inception6(768)
        self.Inception7_1=Inception7(1280)
        self.Inception7_2=Inception7(2048)#Bx8x8x2048
        
        self.pool2=nn.MaxPool2d(8,1)
        self.drop=nn.Dropout(0.4)
        self.conv6=nn.Conv2d(2048,num_classes,1,1)
        self.bn6=nn.BatchNorm2d(num_classes,eps=0.001)
        
    def forward(self,x):
        output=self.conv1(x)
        output=F.relu(self.bn1(output))
        output=self.conv2(output)
        output=F.relu(self.bn2(output))
        output=self.conv3(output)
        output=F.relu(self.bn3(output))
        output=self.pool(output)
        output=self.conv4(output)
        output=F.relu(self.bn4(output))
        output=self.conv5(output)
        output=F.relu(self.bn5(output))
      
        output=self.pool(output)
     
        output=self.Inception1_1(output)
      
        output=self.Inception1_2(output)
     
        output=self.Inception1_3(output)
  
        output=self.Inception2_1(output)
   
        output=self.Inception3_1(output)
        
        output=self.Inception4_1(output)
       
        output=self.Inception4_2(output)
    
        output=self.Inception5_1(output)
     
        output=self.Inception6_1(output)
       
        output=self.Inception7_1(output)
     
        output=self.Inception7_2(output)
    
        
        output=self.drop(self.pool2(output))
        output=self.conv6(output)
        output=F.relu(self.bn6(output))#为什么最后一层不能使用激活函数
        #print(output.shape)
        output=output.view(output.size(0),-1)#4x10
        return output
             
        
class Inception1(nn.Module):
    
    def __init__(self,input_channels,pool_output_channels):#输入的通道数和池化分支的输出通道数
        super(Inception1,self).__init__()
        self.conv1_1=nn.Conv2d(input_channels,64,1,stride=1,bias=False)
        self.bn1_1=nn.BatchNorm2d(64,eps=0.001)
        
        self.conv2_1=nn.Conv2d(input_channels,48,1,stride=1,bias=False)
        self.bn2_1=nn.BatchNorm2d(48,eps=0.001)
        self.conv2_2=nn.Conv2d(48,64,5,stride=1,padding=2,bias=False)
        self.bn2_2=nn.BatchNorm2d(64,eps=0.001)
        
        self.conv3_1=nn.Conv2d(input_channels,64,1,1,bias=False)
        self.bn3_1=nn.BatchNorm2d(64,eps=0.001)
        self.conv3_2=nn.Conv2d(64,96,3,stride=1,padding=1,bias=False)
        self.bn3_2=nn.BatchNorm2d(96,eps=0.001)
        self.conv3_3=nn.Conv2d(96,96,3,stride=1,padding=1,bias=False)
        self.bn3_3=nn.BatchNorm2d(96,eps=0.001)
        
        self.pool=nn.AvgPool2d(3,stride=1,padding=1)
        self.conv4_1=nn.Conv2d(input_channels,pool_output_channels,1,1,bias=False)
        self.bn4_1=nn.BatchNorm2d(pool_output_channels,eps=0.001)
        
    def forward(self,x):
        input1=self.conv1_1(x)
        input1=self.bn1_1(input1)
        input1=F.relu(input1)
        
        input2=self.conv2_1(x)
        input2=self.bn2_1(input2)
        input2=F.relu(input2)
        input2=self.conv2_2(input2)
        input2=self.bn2_2(input2)
        input2=F.relu(input2)
        
        input3=self.conv3_1(x)
        input3=self.bn3_1(input3)
        input3=F.relu(input3)
        input3=self.conv3_2(input3)
        input3=self.bn3_2(input3)
        input3=F.relu(input3)
        input3=self.conv3_3(input3)
        input3=self.bn3_3(input3)
        input3=F.relu(input3)
        
        input4=self.pool(x)
        input4=self.conv4_1(input4)
        input4=self.bn4_1(input4)
        input4=F.relu(input4)
    
        output=[input1,input2,input3,input4]
        return torch.cat(output,1) #BxCxHxW
        
        
class Inception2(nn.Module): #会改变特征图的尺度
    
    def __init__(self,input_channels):
        super(Inception2,self).__init__()
        self.conv1_1=nn.Conv2d(input_channels,384,3,stride=2,bias=False)
        self.bn1_1=nn.BatchNorm2d(384,eps=0.001)
        
        self.conv2_1=nn.Conv2d(input_channels,64,1,stride=1,bias=False)
        self.bn2_1=nn.BatchNorm2d(64,eps=0.001)
        self.conv2_2=nn.Conv2d(64,96,3,stride=1,padding=1,bias=False)
        self.bn2_2=nn.BatchNorm2d(96,eps=0.001)
        self.conv2_3=nn.Conv2d(96,96,3,stride=2,bias=False)
        self.bn2_3=nn.BatchNorm2d(96,eps=0.001)
        
        self.pool=nn.MaxPool2d(3,2)
        
    def forward(self,x):
        input1=self.conv1_1(x)
        input1=F.relu(self.bn1_1(input1))
        
        input2=self.conv2_1(x)
        input2=F.relu(self.bn2_1(input2))
        input2=self.conv2_2(input2)
        input2=F.relu(self.bn2_2(input2))
        input2=self.conv2_3(input2)
        input2=F.relu(self.bn2_3(input2))
        
        input3=self.pool(x)
        
        output=[input1,input2,input3]
        return torch.cat(output,1)
      
        
class Inception3(nn.Module):
        
    def __init__(self,input_channels):
        super(Inception3,self).__init__()
        self.conv1_1=nn.Conv2d(input_channels,192,1,stride=1,bias=False)
        self.bn1_1=nn.BatchNorm2d(192,eps=0.001)
        
        self.conv2_1=nn.Conv2d(input_channels,128,1,stride=1,bias=False)
        self.bn2_1=nn.BatchNorm2d(128,eps=0.001)
        self.conv2_2=nn.Conv2d(128,128,(1,7),(1,1),(0,3),bias=False)
        self.bn2_2=nn.BatchNorm2d(128,eps=0.001)
        self.conv2_3=nn.Conv2d(128,192,(7,1),(1,1),(3,0),bias=False)
        self.bn2_3=nn.BatchNorm2d(192,eps=0.001)
        
        self.conv3_1=nn.Conv2d(input_channels,128,1,1,bias=False)
        self.bn3_1=nn.BatchNorm2d(128,eps=0.001)
        self.conv3_2=nn.Conv2d(128,128,(7,1),(1,1),(3,0),bias=False)
        self.bn3_2=nn.BatchNorm2d(128,eps=0.001)
        self.conv3_3=nn.Conv2d(128,128,(1,7),(1,1),(0,3),bias=False)
        self.bn3_3=nn.BatchNorm2d(128,eps=0.001)
        self.conv3_4=nn.Conv2d(128,128,(7,1),(1,1),(3,0),bias=False)
        self.bn3_4=nn.BatchNorm2d(128,eps=0.001)
        self.conv3_5=nn.Conv2d(128,192,(1,7),(1,1),(0,3),bias=False)
        self.bn3_5=nn.BatchNorm2d(192,eps=0.001)
    
        self.pool=nn.AvgPool2d(3,1,padding=1)
        self.conv4_1=nn.Conv2d(input_channels,192,1,1,bias=False)
        self.bn4_1=nn.BatchNorm2d(192,eps=0.001)
        
    def forward(self,x):
        input1=self.conv1_1(x)
        input1=F.relu(self.bn1_1(input1))
        
        input2=self.conv2_1(x)
        input2=F.relu(self.bn2_1(input2))
        input2=self.conv2_2(input2)
        input2=F.relu(self.bn2_2(input2))
        input2=self.conv2_3(input2)
        input2=F.relu(self.bn2_3(input2))
        
        input3=self.conv3_1(x)
        input3=F.relu(self.bn3_1(input3))
        input3=self.conv3_2(input3)
        input3=F.relu(self.bn3_2(input3))
        input3=self.conv3_3(input3)
        input3=F.relu(self.bn3_3(input3))
        input3=self.conv3_4(input3)
        input3=F.relu(self.bn3_4(input3))
        input3=self.conv3_5(input3)
        input3=F.relu(self.bn3_5(input3))
        
        input4=self.pool(x)
        input4=self.conv4_1(input4)
        input4=F.relu(self.bn4_1(input4))
        
        output=[input1,input2,input3,input4]
        return torch.cat(output,1)
        
        
class Inception4(nn.Module):
        
    def __init__(self,input_channels):
        super(Inception4,self).__init__()
        self.conv1_1=nn.Conv2d(input_channels,192,1,stride=1,bias=False)
        self.bn1_1=nn.BatchNorm2d(192,eps=0.001)
        
        self.conv2_1=nn.Conv2d(input_channels,160,1,stride=1,bias=False)
        self.bn2_1=nn.BatchNorm2d(160,eps=0.001)
        self.conv2_2=nn.Conv2d(160,160,(1,7),(1,1),(0,3),bias=False)
        self.bn2_2=nn.BatchNorm2d(160,eps=0.001)
        self.conv2_3=nn.Conv2d(160,192,(7,1),(1,1),(3,0),bias=False)
        self.bn2_3=nn.BatchNorm2d(192,eps=0.001)
        
        self.conv3_1=nn.Conv2d(input_channels,160,1,1,bias=False)
        self.bn3_1=nn.BatchNorm2d(160,eps=0.001)
        self.conv3_2=nn.Conv2d(160,160,(7,1),(1,1),(3,0),bias=False)
        self.bn3_2=nn.BatchNorm2d(160,eps=0.001)
        self.conv3_3=nn.Conv2d(160,160,(1,7),(1,1),(0,3),bias=False)
        self.bn3_3=nn.BatchNorm2d(160,eps=0.001)
        self.conv3_4=nn.Conv2d(160,160,(7,1),(1,1),(3,0),bias=False)
        self.bn3_4=nn.BatchNorm2d(160,eps=0.001)
        self.conv3_5=nn.Conv2d(160,192,(1,7),(1,1),(0,3),bias=False)
        self.bn3_5=nn.BatchNorm2d(192,eps=0.001)
    
        self.pool=nn.AvgPool2d(3,1,padding=1)
        self.conv4_1=nn.Conv2d(input_channels,192,1,1,bias=False)
        self.bn4_1=nn.BatchNorm2d(192,eps=0.001)
        
    def forward(self,x):
        input1=self.conv1_1(x)
        input1=F.relu(self.bn1_1(input1))
        
        input2=self.conv2_1(x)
        input2=F.relu(self.bn2_1(input2))
        input2=self.conv2_2(input2)
        input2=F.relu(self.bn2_2(input2))
        input2=self.conv2_3(input2)
        input2=F.relu(self.bn2_3(input2))
        
        input3=self.conv3_1(x)
        input3=F.relu(self.bn3_1(input3))
        input3=self.conv3_2(input3)
        input3=F.relu(self.bn3_2(input3))
        input3=self.conv3_3(input3)
        input3=F.relu(self.bn3_3(input3))
        input3=self.conv3_4(input3)
        input3=F.relu(self.bn3_4(input3))
        input3=self.conv3_5(input3)
        input3=F.relu(self.bn3_5(input3))
        
        input4=self.pool(x)
        input4=self.conv4_1(input4)
        input4=F.relu(self.bn4_1(input4))
        
        output=[input1,input2,input3,input4]
        return torch.cat(output,1)
        
class Inception5(nn.Module):
        
    def __init__(self,input_channels):
        super(Inception5,self).__init__()
        self.conv1_1=nn.Conv2d(input_channels,192,1,stride=1,bias=False)
        self.bn1_1=nn.BatchNorm2d(192,eps=0.001)
        
        self.conv2_1=nn.Conv2d(input_channels,192,1,stride=1,bias=False)
        self.bn2_1=nn.BatchNorm2d(192,eps=0.001)
        self.conv2_2=nn.Conv2d(192,192,(1,7),(1,1),(0,3),bias=False)
        self.bn2_2=nn.BatchNorm2d(192,eps=0.001)
        self.conv2_3=nn.Conv2d(192,192,(7,1),(1,1),(3,0),bias=False)
        self.bn2_3=nn.BatchNorm2d(192,eps=0.001)
        
        self.conv3_1=nn.Conv2d(input_channels,192,1,1,bias=False)
        self.bn3_1=nn.BatchNorm2d(192,eps=0.001)
        self.conv3_2=nn.Conv2d(192,192,(7,1),(1,1),(3,0),bias=False)
        self.bn3_2=nn.BatchNorm2d(192,eps=0.001)
        self.conv3_3=nn.Conv2d(192,192,(1,7),(1,1),(0,3),bias=False)
        self.bn3_3=nn.BatchNorm2d(192,eps=0.001)
        self.conv3_4=nn.Conv2d(192,192,(7,1),(1,1),(3,0),bias=False)
        self.bn3_4=nn.BatchNorm2d(192,eps=0.001)
        self.conv3_5=nn.Conv2d(192,192,(1,7),(1,1),(0,3),bias=False)
        self.bn3_5=nn.BatchNorm2d(192,eps=0.001)
    
        self.pool=nn.AvgPool2d(3,1,padding=1)
        self.conv4_1=nn.Conv2d(input_channels,192,1,1,bias=False)
        self.bn4_1=nn.BatchNorm2d(192,eps=0.001)
        
    def forward(self,x):
        input1=self.conv1_1(x)
        input1=F.relu(self.bn1_1(input1))
        
        input2=self.conv2_1(x)
        input2=F.relu(self.bn2_1(input2))
        input2=self.conv2_2(input2)
        input2=F.relu(self.bn2_2(input2))
        input2=self.conv2_3(input2)
        input2=F.relu(self.bn2_3(input2))
        
        input3=self.conv3_1(x)
        input3=F.relu(self.bn3_1(input3))
        input3=self.conv3_2(input3)
        input3=F.relu(self.bn3_2(input3))
        input3=self.conv3_3(input3)
        input3=F.relu(self.bn3_3(input3))
        input3=self.conv3_4(input3)
        input3=F.relu(self.bn3_4(input3))
        input3=self.conv3_5(input3)
        input3=F.relu(self.bn3_5(input3))
        
        input4=self.pool(x)
        input4=self.conv4_1(input4)
        input4=F.relu(self.bn4_1(input4))
        
        output=[input1,input2,input3,input4]
        return torch.cat(output,1)        
class Inception6(nn.Module): #会改变特征图尺度
    def __init__(self,input_channels):
        super(Inception6,self).__init__()
        self.conv1_1=nn.Conv2d(input_channels,192,1,1,bias=False)
        self.bn1_1=nn.BatchNorm2d(192,eps=0.001)
        self.conv1_2=nn.Conv2d(192,320,3,2,bias=False)
        self.bn1_2=nn.BatchNorm2d(320,eps=0.001)
        
        self.conv2_1=nn.Conv2d(input_channels,192,1,stride=1,bias=False)
        self.bn2_1=nn.BatchNorm2d(192,eps=0.001)
        self.conv2_2=nn.Conv2d(192,192,(1,7),(1,1),(0,3),bias=False)
        self.bn2_2=nn.BatchNorm2d(192,eps=0.001)
        self.conv2_3=nn.Conv2d(192,192,(7,1),(1,1),(3,0),bias=False)
        self.bn2_3=nn.BatchNorm2d(192,eps=0.001)
        self.conv2_4=nn.Conv2d(192,192,3,2,bias=False)
        self.bn2_4=nn.BatchNorm2d(192,eps=0.001)
        
        self.pool=nn.MaxPool2d(3,2)
     
    def forward(self,x):
        
        input1=self.conv1_1(x)
        input1=F.relu(self.bn1_1(input1))
        input1=self.conv1_2(input1)
        input1=F.relu(self.bn1_2(input1))
        
        input2=self.conv2_1(x)
        input2=F.relu(self.bn2_1(input2))
        input2=self.conv2_2(input2)
        input2=F.relu(self.bn2_2(input2))
        input2=self.conv2_3(input2)
        input2=F.relu(self.bn2_3(input2))
        input2=self.conv2_4(input2)
        input2=F.relu(self.bn2_4(input2))
        
        input3=self.pool(x)
        
        output=[input1,input2,input3]
        return torch.cat(output,1)
    
class Inception7(nn.Module):
    
    def __init__(self,input_channels):
        super(Inception7,self).__init__()
        self.conv1_1=nn.Conv2d(input_channels,320,1,1)
        self.bn1_1=nn.BatchNorm2d(320,eps=0.001)
        
        self.conv2_1=nn.Conv2d(input_channels,384,1,1)
        self.bn2_1=nn.BatchNorm2d(384,eps=0.001)
        self.conv2_2=nn.Conv2d(384,384,(1,3),(1,1),(0,1))
        self.bn2_2=nn.BatchNorm2d(384,eps=0.001)
        self.conv2_3=nn.Conv2d(384,384,(3,1),(1,1),(1,0))
        self.bn2_3=nn.BatchNorm2d(384,eps=0.001)
        
        self.conv3_1=nn.Conv2d(input_channels,448,1,1)
        self.bn3_1=nn.BatchNorm2d(448,eps=0.001)
        self.conv3_2=nn.Conv2d(448,384,3,1,1)
        self.bn3_2=nn.BatchNorm2d(384,eps=0.001)
        self.conv3_3=nn.Conv2d(384,384,(1,3),(1,1),(0,1))
        self.bn3_3=nn.BatchNorm2d(384,eps=0.001)
        self.conv3_4=nn.Conv2d(384,384,(3,1),(1,1),(1,0))
        self.bn3_4=nn.BatchNorm2d(384,eps=0.001)
        
        self.pool=nn.AvgPool2d(3,1,1)
        self.conv4_1=nn.Conv2d(input_channels,192,1,1)
        self.bn4_1=nn.BatchNorm2d(192,eps=0.001)
    
    def forward(self,x):
        
        input1=self.conv1_1(x)
        input1=F.relu(self.bn1_1(input1))
        
        input2=self.conv2_1(x)
        input2=F.relu(self.bn2_1(input2))
        input2_1=self.conv2_2(input2)
        input2_1=F.relu(self.bn2_2(input2))
        input2_2=self.conv2_3(input2)
        input2_2=F.relu(self.bn2_3(input2))
        input2=torch.cat([input2_1,input2_2],1)
        
        input3=self.conv3_1(x)
        input3=F.relu(self.bn3_1(input3))
        input3=self.conv3_2(input3)
        input3=F.relu(self.bn3_2(input3))
        input3=self.conv3_3(input3)
        input3_1=F.relu(self.bn3_3(input3))
        input3=self.conv3_4(input3)
        input3_2=F.relu(self.bn3_4(input3))
        input3=torch.cat([input3_1,input3_2],1)
        
        input4=self.pool(x)
        input4=self.conv4_1(input4)
        input4=F.relu(self.bn4_1(input4))
        
        output=[input1,input2,input3,input4]
        return torch.cat(output,1)
    
    
    
'''test_net = InceptionNet(10)
test_net.eval()
test_x = Variable(torch.zeros(1, 3, 299, 299))
test_y = test_net(test_x)
print('output: {}'.format(test_y.shape))'''     
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    