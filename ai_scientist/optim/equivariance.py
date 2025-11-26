"""Equivariance and Geometric Deep Learning modules (Phase 1.2).

This module provides geometric encoders that can be used to enforce or learn
SE(3) symmetries (rotation/translation) and Permutation invariance.
Since e3nn is not available, we implement PointNet with Spatial Transformer (T-Net)
alignment to achieve approximate SE(3) invariance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TNet3d(nn.Module):
    """Spatial Transformer Network for 3D Point Clouds.
    
    Learns a 3x3 transformation matrix to canonically align the input point cloud,
    making the downstream network invariant to rigid rotations (if the T-Net learns to align).
    """
    def __init__(self, input_dim: int = 3):
        super().__init__()
        self.input_dim = input_dim
        
        # MLP to extract global features from points
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, input_dim * input_dim)
        
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        
        # Identity initialization
        self.identity = torch.eye(input_dim, dtype=torch.float32).view(1, input_dim * input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Point cloud (Batch, input_dim, N_points)
        Returns:
            trans: Transformation matrix (Batch, input_dim, input_dim)
        """
        bs = x.size(0)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Global pooling
        x = torch.max(x, 2, keepdim=False)[0]
        
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        
        # Add identity to start from a stable state
        # Move identity to same device as x
        idt = self.identity.to(x.device).repeat(bs, 1)
        x = x + idt
        
        x = x.view(-1, self.input_dim, self.input_dim)
        return x


class PointNetEncoder(nn.Module):
    """Geometric Encoder using PointNet architecture.
    
    Provides:
    1. Permutation Invariance (via symmetric MaxPool).
    2. SE(3) Invariance (approximate, via T-Net alignment).
    """
    
    def __init__(self, embedding_dim: int = 1024, align_input: bool = True):
        super().__init__()
        self.align_input = align_input
        
        if align_input:
            self.input_transform = TNet3d(input_dim=3)
        
        # Point-wise MLPs (implemented as Conv1d with kernel=1)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, embedding_dim, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(embedding_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Point cloud (Batch, 3, N_points) - Note channel first!
        Returns:
            global_feat: (Batch, embedding_dim)
        """
        if self.align_input:
            # T-Net expects (B, 3, N)
            transform = self.input_transform(x)
            # Apply transform: (B, 3, 3) x (B, 3, N) -> (B, 3, N)
            x = torch.bmm(transform, x)
            
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Symmetric function: Max Pooling over points
        x = torch.max(x, 2, keepdim=False)[0]
        
        return x
