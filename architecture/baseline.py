"""
Fixed baseline architecture: a small, completely standard CNN.

This is the comparison point for every experiment in the project. Do not
edit this file to make a new mechanism look better — if the baseline
needs to change, that's a separate, explicitly-labeled experiment (see
root skills.md, "Dangerous Areas").
"""

import torch.nn as nn


class SmallCNN(nn.Module):
    """
    A small CNN for MNIST-sized (1x28x28) images. Two conv blocks, one
    hidden fully-connected layer, then a classifier head.

    `activation_fn` is a factory (a zero-arg callable returning an
    nn.Module) so the same architecture can be reused with either a
    standard ReLU or a sparse activation like kWTA — this is what makes
    the baseline and the experimental model directly comparable: identical
    architecture, only the activation differs.
    """

    def __init__(self, num_classes: int = 10, activation_fn=nn.ReLU):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            activation_fn(),
            nn.MaxPool2d(2),  # 28x28 -> 14x14

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            activation_fn(),
            nn.MaxPool2d(2),  # 14x14 -> 7x7
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            activation_fn(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
