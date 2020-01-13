#_*_ coding:utf-8 _*_
import torch
import model
from model import InceptionNet as model
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import mydataset
#加载测试数据
transform=transforms.Compose([transforms.Scale(299),transforms.ToTensor()])

testdata=traindata=mydataset.Mydataset("/app/datas/6/mini-imagenet/images/",data_type="test",transform=transform)

testloader=DataLoader(testdata,batch_size=4,shuffle=True,num_workers=2)


#加载模型
num_class=64
net=model(num_class)
net.load_state_dict(torch.load("model3_inception.pth"))
net.eval()
all=0
correct=0
for data in testloader:
    img,label=data
    img=Variable(img)#Variable类型
    _,predicted=torch.max(net(img).data,1)#含有两个张量，前者是每行最大的值（FloatTensor），后者是对应值的下标（LongTensor）
    all+=label.size(0)
    #LongTensor转FloatTensor
    correct+=(label==predicted).sum()#计算与标签相同的个数
    
print("模型的准确率是：",float(correct/all))
    