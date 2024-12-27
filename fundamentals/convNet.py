import torch
from torch import nn
import torch.nn.functional as F

class convNet(nn.Module):
    def __init__(self):
        super(convNet,self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=3,padding='same')
        self.pool1 = nn.MaxPool2d(2,stride=2)
        self.conv2 = nn.Conv2d(10,10,kernel_size=3,padding="same")
        self.pool2 = nn.MaxPool2d(2,stride=2)
        self.fc = nn.Linear(640,10,bias=False)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        logits = self.fc(torch.flatten(x))
        return logits

if __name__ == "__main__":
    img = torch.randn((1,3,32,32))
    cnet = convNet()
    logits = cnet(img)
    print(logits.size())
