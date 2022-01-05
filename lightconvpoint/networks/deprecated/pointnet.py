import torch
import torch.nn as nn
import torch.nn.functional as F

class Pointnet(nn.Module):

    def __init__(self, in_channels,
            out_channels,
            hidden_dim, segmentation=False):
        super().__init__()
        
        self.fc_in = nn.Conv1d(in_channels, 2*hidden_dim, 1)
        self.fc_0 = nn.Conv1d(2*hidden_dim, hidden_dim, 1)
        self.fc_1 = nn.Conv1d(2*hidden_dim, hidden_dim, 1)
        self.fc_2 = nn.Conv1d(2*hidden_dim, hidden_dim, 1)
        self.fc_3 = nn.Conv1d(2*hidden_dim, hidden_dim, 1)

        self.segmentation=segmentation

        if segmentation:
            self.fc_out = nn.Conv1d(2*hidden_dim, out_channels, 1)
        else:
            self.fc_out = nn.Linear(hidden_dim, out_channels)

        self.activation = nn.ReLU()

    def forward(self, x):

        x = self.fc_in(x)

        x = self.fc_0(self.activation(x))
        x_pool = torch.max(x, dim=2, keepdim=True)[0].expand_as(x)
        x = torch.cat([x, x_pool], dim=1)
        
        x = self.fc_1(self.activation(x))
        x_pool = torch.max(x, dim=2, keepdim=True)[0].expand_as(x)
        x = torch.cat([x, x_pool], dim=1)

        x = self.fc_2(self.activation(x))
        x_pool = torch.max(x, dim=2, keepdim=True)[0].expand_as(x)
        x = torch.cat([x, x_pool], dim=1)

        x = self.fc_3(self.activation(x))

        if self.segmentation:
            x_pool = torch.max(x, dim=2, keepdim=True)[0].expand_as(x)
            x = torch.cat([x, x_pool], dim=1)
        else:
            x = torch.max(x, dim=2)[0]
        
        x = self.fc_out(x)

        return x


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, hidden_dim):
        super().__init__()

        # Submodules
        self.fc_0 = nn.Conv1d(in_channels, hidden_dim, 1)
        self.fc_1 = nn.Conv1d(hidden_dim, out_channels, 1)
        self.activation = nn.ReLU()

        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels,1)
        else:
            self.shortcut = nn.Identity()

        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        x_short = self.shortcut(x)
        x = self.fc_0(x)
        x = self.fc_1(self.activation(x))
        x = self.activation(x + x_short)
        return x



class ResidualPointnet(nn.Module):
    ''' PointNet-based encoder network with ResNet blocks.
    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, in_channels, out_channels, hidden_dim, segmentation=False):
        super().__init__()

        self.fc_in = nn.Conv1d(in_channels, 2*hidden_dim, 1)
        self.block_0 = ResidualBlock(2*hidden_dim, hidden_dim, hidden_dim)
        self.block_1 = ResidualBlock(2*hidden_dim, hidden_dim, hidden_dim)
        self.block_2 = ResidualBlock(2*hidden_dim, hidden_dim, hidden_dim)
        self.block_3 = ResidualBlock(2*hidden_dim, hidden_dim, hidden_dim)
        self.block_4 = ResidualBlock(2*hidden_dim, hidden_dim, hidden_dim)

        self.segmentation = segmentation
        if self.segmentation:
            self.fc_out = nn.Conv1d(2*hidden_dim, out_channels, 1)
        else:
            self.fc_out = nn.Linear(hidden_dim, out_channels)


    def forward(self, x):

        x = self.fc_in(x)
        
        x = self.block_0(x)
        x_pool = torch.max(x, dim=2, keepdim=True)[0].expand_as(x)
        x = torch.cat([x, x_pool], dim=1)
        
        x = self.block_1(x)
        x_pool = torch.max(x, dim=2, keepdim=True)[0].expand_as(x)
        x = torch.cat([x, x_pool], dim=1)
        
        x = self.block_2(x)
        x_pool = torch.max(x, dim=2, keepdim=True)[0].expand_as(x)
        x = torch.cat([x, x_pool], dim=1)
        
        x = self.block_3(x)
        x_pool = torch.max(x, dim=2, keepdim=True)[0].expand_as(x)
        x = torch.cat([x, x_pool], dim=1)

        x = self.block_4(x)

        if self.segmentation:
            x_pool = torch.max(x, dim=2, keepdim=True)[0].expand_as(x)
            x = torch.cat([x, x_pool], dim=1)
        else:
            x = torch.max(x, dim=2)[0]

        x = self.fc_out(x)

        return x