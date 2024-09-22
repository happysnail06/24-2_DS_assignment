import torch
from torch import nn
from torch.nn import functional as F

class CNN(nn.Module):
    def __init__(self, input_channels: int = 1, num_classes: int = 3) -> None:
        """
            65 × 65 images
            an input layer, two convolutional layers with 3 × 3 filters, 
            a max pooling layer, 
            two dropout layers with rates 0.25 and 0.5, 
            a fully connected layer
            and finally an output layer.
        """
        super(CNN, self).__init__()
        
        # First Convolutional Layer
        self.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=32,
            kernel_size=5,
            padding=2  # To maintain spatial dimensions (same padding)
        )
        
        # Second Convolutional Layer
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=5,
            padding=2  # To maintain spatial dimensions
        )
        
        # Max Pooling Layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Dropout Layers
        self.dropout1 = nn.Dropout(p=0.25)  # First Dropout
        self.dropout2 = nn.Dropout(p=0.5)   # Second Dropout
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(in_features=64 * 32 * 32, out_features=128)
        
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() == 4:
            if x.size(1) != self.conv1.in_channels:
                x = x.permute(0, 3, 1, 2)
        
        x = self.conv1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        
        x = self.pool(x)
        
        x = self.dropout1(x)
        
        x = x.contiguous().view(x.size(0), -1)
        
        x = self.fc1(x)
        x = F.relu(x)

        x = self.dropout2(x)
        
        x = self.fc2(x)
        
        return x
