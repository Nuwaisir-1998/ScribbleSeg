# CNN model
from config import Config
import torch.nn as nn

class MyNet(nn.Module):
    def __init__(self,input_dim,last_layer_channel_count):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, Config.intermediate_channels, kernel_size=3, stride=1, padding=1 )
        self.bn1 = nn.BatchNorm2d(Config.intermediate_channels)
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        for i in range(Config.nConv-1):
            self.conv2.append( nn.Conv2d(Config.intermediate_channels, Config.intermediate_channels, kernel_size=3, stride=1, padding=1 ) )
            self.bn2.append( nn.BatchNorm2d(Config.intermediate_channels) )

        r = last_layer_channel_count

        print('last layer size:', r)
        self.conv3 = nn.Conv2d(Config.intermediate_channels, r, kernel_size=1, stride=1, padding=0 )
        self.bn3 = nn.BatchNorm2d(r)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu( x )
        x = self.bn1(x)
        for i in range(Config.nConv-1):
            x = self.conv2[i](x)
            x = F.relu( x )
            x = self.bn2[i](x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x