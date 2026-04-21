import torch.nn as nn


class ObstacleClassifier(nn.Module):
    """Small CNN: 32x32 grayscale crop -> {0: decoy, 1: ground, 2: flying}."""

    def __init__(self):
        super().__init__()
        # three conv blocks that progressively extract features and shrink spatial size,
        # then flatten and classify into one of three obstacle types
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> 16x16
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> 8x8
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),  # -> 4x4, handles any input size
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),  # prevents overfitting on the small training set
            nn.Linear(128, 3),  # output: 0=decoy, 1=ground, 2=flying
        )

    def forward(self, x):
        return self.net(x)
