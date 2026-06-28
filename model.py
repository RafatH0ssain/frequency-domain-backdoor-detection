import torch
import torch.nn as nn

class LightweightCNN(nn.Module):
    """
    A small CNN designed to mirror the structural properties of the 
    lightweight models reviewed in the literature (Ali et al. [1], 
    Rakhmetova et al. [2]): aggressive max-pooling, limited filter count, 
    and large strides that create spatial blind spots.
    """
    def __init__(self, num_classes=10):
        super(LightweightCNN, self).__init__()
        
        # Block 1 — 32 filters, mirrors Ali et al. [1] exactly
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 32x32 → 16x16
        
        # Block 2 — 64 filters, still lightweight
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 16x16 → 8x8
        
        # Block 3 — 64 filters, further compression
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 8x8 → 4x4
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
        # Classifier head
        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.relu(self.conv3(x))
        x = self.pool3(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    model = LightweightCNN()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    # Quick shape test
    dummy = torch.randn(1, 3, 32, 32)
    out = model(dummy)
    print(f"Output shape: {out.shape}")  # should be [1, 10]