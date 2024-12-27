import torch
from torch import nn
import torch.nn.functional as F

class batchNorm2d(nn.Module):
    def __init__(self,n_features,momentum=0.1):
        super(batchNorm2d, self).__init__()
        self.mean = torch.zeros(n_features)
        self.var = torch.zeros(n_features)
        self.momentum = momentum
        self.beta = torch.ones((1,n_features,1,1))
        self.gamma = torch.zeros((1,n_features,1,1))
    
    def forward(self,x):
        current_mean = torch.mean(x,dim=(0,2,3),keepdim=True)
        current_var = torch.var(x,dim=(0,2,3),keepdim=True)
        self.mean = (1-self.momentum) * self.mean + self.momentum * current_mean
        self.var = (1-self.momentum) * self.var + self.momentum * current_var
        normalized = (x - current_mean)/current_var
        output = self.beta* normalized+ self.gamma
        return output


class convNet(nn.Module):
    def __init__(self):
        super(convNet,self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=3,padding='same')
        self.pool1 = nn.MaxPool2d(2,stride=2)
        self.bn1 = batchNorm2d(10)
        self.conv2 = nn.Conv2d(10,10,kernel_size=3,padding="same")
        self.pool2 = nn.MaxPool2d(2,stride=2)
        self.fc = nn.Linear(640,10,bias=False)

    def forward(self,x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        logits = self.fc(torch.flatten(x,start_dim=1))
        return logits

if __name__ == "__main__":
    img = torch.randn((2,3,32,32))
    cnet = convNet()
    logits = cnet(img)
    print(logits.size())
