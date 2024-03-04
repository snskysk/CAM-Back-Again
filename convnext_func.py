import torch.nn as nn
import torch.nn.functional as F

class Net2Head(nn.Module):
    def __init__(self, class_n, unit_n, size):
        super().__init__() 
        self.unit_n = unit_n
        self.size = size
        self.fc = nn.Linear(unit_n, class_n)
        self.cam = nn.AvgPool2d(size,size)
    
    def forward(self,x):     
        x = self.cam(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.softmax(x, dim=1)
    
