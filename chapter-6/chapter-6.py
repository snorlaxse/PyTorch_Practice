# -*- coding: utf-8 -*-
# 6.4.1 torch和torchvision
import torch
import torchvision
from torchvision import datasets, transforms  # torchvision包的主要功能是实现数据的处理、导入和预览
from torch.autograd import Variable
import matplotlib.pyplot as plt

# %matplotlib inline

transform=transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])]) 

data_train = datasets.MNIST(root = "./data/",  # .表示保存在当前文件夹的路径下
                            transform=transform,
                            train = True,   # 载入训练集
                            download = True)

data_test = datasets.MNIST(root="./data/",
                           transform=transform,
                           train = False   # 载入测试集
                          )

data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                                batch_size = 64,
                                                shuffle = True,
                                               )

data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                               batch_size = 64,
                                               shuffle = True)

images, labels = next(iter(data_loader_train))

img = torchvision.utils.make_grid(images)   # (batch_size,channel,height,weight) ->> (channel,height,weight)
img = img.numpy().transpose(1,2,0)  # 完成数据类型的转换和数据维度的交换  (channel,height,weight) ->> (height, weight, channel)

std = [0.5,0.5,0.5]
mean = [0.5,0.5,0.5]
img = img*std+mean
print([labels[i] for i in range(64)])
plt.imshow(img)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1=torch.nn.Sequential(
            torch.nn.Conv2d(1,64,kernel_size=3,stride=1,padding=1),  # 卷积层
            torch.nn.ReLU(), 
            torch.nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=2,kernel_size=2))
        self.dense=torch.nn.Sequential(
            torch.nn.Linear(14*14*128,1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(1024, 10))
        
    def forward(self, x):
        x = self.conv1(x)   # 卷积处理
        x = x.view(-1, 14*14*128)  # 对参数实现扁平化，make sure维度匹配
        x = self.dense(x) # 全连接层进行分类
        return x

model = Model()
cost = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

print(model)

n_epochs = 5

for epoch in range(n_epochs):
    running_loss = 0.0
    running_correct = 0
    print("Epoch {}/{}".format(epoch, n_epochs))
    print("-"*10)
    
    for data in data_loader_train:
        print(data)
        X_train, y_train = data
        X_train, y_train = Variable(X_train), Variable(y_train)
        outputs = model(X_train)
        _,pred = torch.max(outputs.data, 1)
        optimizer.zero_grad()
        loss = cost(outputs, y_train)
        
        loss.backward()
        optimizer.step()     
        running_loss += loss.data
        running_correct += torch.sum(pred == y_train.data)

    testing_correct = 0    
    
    for data in data_loader_test:
        X_test, y_test = data
        X_test, y_test = Variable(X_test), Variable(y_test)
        outputs = model(X_test)
        _, pred = torch.max(outputs.data, 1)
        testing_correct += torch.sum(pred == y_test.data)
    print("Loss is:{:.4f}, Train Accuracy is:{:.4f}%, Test Accuracy is:{:.4f}".format(
        running_loss/len(data_train),100*running_correct/len(data_train),100*testing_correct/len(data_test)))