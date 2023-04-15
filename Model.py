import torch
from torch import nn

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(5, 5)),   
            nn.Tanh(),                                                      
            nn.AvgPool2d(2, 2),                                            

            nn.Conv2d(in_channels=4, out_channels=12, kernel_size=(5, 5)), 
            nn.Tanh(),
            nn.AvgPool2d(2, 2) 
        )

        self.linear = nn.Sequential(
            nn.Linear(4*4*12,10)
        )
    
    def forward(self, x):
        x = self.convs(x)
        x = torch.flatten(x, 1)

        return self.linear(x)