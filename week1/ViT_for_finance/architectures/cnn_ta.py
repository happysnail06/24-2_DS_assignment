import torch
from torch import nn
from torch.nn import functional as F


class CNN(nn.Module):
    def __init__(self, input_channels: int = 65, num_classes: int = 2) -> None:
        """
            65 × 65 images
            an input layer, two convolutional layers with 3 × 3 filters, 
            a max pooling layer, 
            two dropout layers with rates 0, 25 and 0.5, 
            a fully connected layer
            and finally an output layer.
        """
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        
        self.dropout1 = nn.Dropout(p=0.25) # from paper
        self.dropout2 = nn.Dropout(p=0.5) # from paper

        self.adaptive_pool = nn.AdaptiveAvgPool2d((16, 16))
        
        self.fc1 = nn.Linear(in_features=64 * 16 * 16, out_features=128)
        self.relu3 = nn.ReLU()
        
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() == 4 and x.size(1) != 65:
            x = x.permute(0, 3, 1, 2)
        
        x = self.conv1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        
        x = self.pool(x)
        
        x = self.dropout1(x)
        
        x = self.adaptive_pool(x)

        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.relu3(x)
        
        x = self.dropout2(x)
        
        x = self.fc2(x)
        return x