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
        self.identity = torch.eye(input_dim, dtype=torch.float32).view(
            1, input_dim * input_dim
        )

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


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    # Explicitly normalize to eliminate numerical drift from floating-point operations.
    # Without this, quaternions may have norm slightly != 1 (e.g., 0.999998 or 1.000002),
    # causing small but compounding errors in the rotation matrix.
    quaternions = quaternions / quaternions.norm(dim=-1, keepdim=True)

    r, i, j, k = torch.unbind(quaternions, -1)
    # For unit quaternions, |q|² = 1, so 2/|q|² = 2.0
    two_s = 2.0

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def random_rotation_matrix(
    batch_size: int, device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """Generate random 3x3 rotation matrices (SO(3)) using Quaternions.

    Replaces QR-based generation to avoid MPS fallback issues.
    Algorithm: Uniformly sample unit quaternions from S^3, then convert to rotation matrices
    (Shoemake, 1992). This allows the entire generation to happen on-device (GPU/MPS)
    without CPU fallback or synchronization.
    """
    # 1. Sample 3 uniform random variables on the device
    # u1, u2, u3 ~ U[0, 1]
    u = torch.rand(batch_size, 3, device=device)
    u1, u2, u3 = u[:, 0], u[:, 1], u[:, 2]

    # 2. Construct Unit Quaternion
    # q = [w, x, y, z]
    # theta1 = 2*pi*u2, theta2 = 2*pi*u3
    # R1 = sqrt(1-u1), R2 = sqrt(u1)

    sqrt_1_minus_u1 = torch.sqrt(1 - u1)
    sqrt_u1 = torch.sqrt(u1)
    theta1 = 2 * torch.pi * u2
    theta2 = 2 * torch.pi * u3

    w = sqrt_1_minus_u1 * torch.sin(theta1)
    x = sqrt_1_minus_u1 * torch.cos(theta1)
    y = sqrt_u1 * torch.sin(theta2)
    z = sqrt_u1 * torch.cos(theta2)

    # Stack to form quaternions (B, 4)
    # Shoemake's convention uses z (sqrt(u1)*cos(theta2)) as the scalar part.
    # Our quaternion_to_matrix expects [real, i, j, k].
    # So we order it: [z, w, x, y]
    quaternions = torch.stack([z, w, x, y], dim=1)

    # 3. Convert Quaternion to Rotation Matrix
    return quaternion_to_matrix(quaternions)
