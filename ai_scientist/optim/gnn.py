"""Graph Neural Network (GNN) modules for Geometric Surrogates (Phase 2.3).

This module implements a custom Graph Neural Network backbone to replace PointNet.
It is designed to work on the toroidal mesh structure of the stellarator surface.
Since torch_geometric is not guaranteed, we implement basic Message Passing layers
using pure PyTorch scatter/gather operations.
"""

import torch
import torch.nn as nn

from ai_scientist.optim import equivariance


class MessagePassingLayer(nn.Module):
    """Basic Message Passing Layer (Graph Convolution).

    Implements a simple message passing scheme:
    h_i' = MLP(h_i || AGG_{j in N(i)} (h_j))
    """

    def __init__(self, in_channels: int, out_channels: int, aggregation: str = "mean"):
        super().__init__()
        self.aggregation = aggregation

        # Message / Update function
        # We concatenate node features with aggregated neighbor features
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 2, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features (Batch, N_nodes, in_channels)
            edge_index: Graph connectivity (2, N_edges).
                        Assumes same graph structure for all items in batch (shared edges),
                        OR we treat batch as separate dimension.

        For efficiency with shared topology (grid), we assume `edge_index` is fixed
        relative to the nodes in the batch.
        """
        batch_size, num_nodes, c_in = x.shape

        # Source and Target indices
        src_idx = edge_index[0]  # (E,)
        dst_idx = edge_index[1]  # (E,)

        # 1. Gather source features
        # x is (B, N, C)
        # We want x_j for every edge.
        # x[:, src_idx, :] -> (B, E, C)
        x_src = x[:, src_idx, :]

        # 2. Aggregate messages at destination
        # We need to scatter_reduce (add/mean) x_src into target nodes.
        # Output container: (B, N, C)

        # PyTorch scatter requires index to be broadcastable.
        # dst_idx is (E,). We need (B, E, C) -> scatter to (B, N, C) along dim 1.
        # dst_idx_expanded: (B, E, C)
        dst_idx_expanded = dst_idx.view(1, -1, 1).expand(batch_size, -1, c_in)

        # Init aggregation tensor
        out_agg = torch.zeros(
            batch_size, num_nodes, c_in, device=x.device, dtype=x.dtype
        )

        # Scatter Add
        out_agg.scatter_add_(1, dst_idx_expanded, x_src)

        # Handle Mean
        if self.aggregation == "mean":
            # Compute degree
            ones = torch.ones((1, dst_idx.size(0), 1), device=x.device, dtype=x.dtype)
            degree = torch.zeros((1, num_nodes, 1), device=x.device, dtype=x.dtype)
            degree.scatter_add_(1, dst_idx.view(1, -1, 1), ones)
            degree = degree.clamp(min=1.0)
            out_agg = out_agg / degree

        # 3. Update
        # Concatenate self features with aggregated neighbor features
        # (B, N, 2*C)
        combined = torch.cat([x, out_agg], dim=-1)

        # Flatten for MLP: (B*N, 2*C)
        combined_flat = combined.view(-1, c_in * 2)
        out = self.mlp(combined_flat)

        # Reshape back: (B, N, out)
        out = out.view(batch_size, num_nodes, -1)

        # Residual connection if shapes match
        if c_in == out.shape[-1]:
            out = out + x

        return out


class ToroidalGridGraph(nn.Module):
    """Helper to manage the toroidal grid graph structure."""

    def __init__(self, n_theta: int, n_zeta: int):
        super().__init__()
        self.n_theta = n_theta
        self.n_zeta = n_zeta
        self.num_nodes = n_theta * n_zeta

        # Precompute edges
        self.register_buffer("edge_index", self._build_edges())

    def _build_edges(self) -> torch.Tensor:
        """Construct edge index for a toroidal grid."""
        # Nodes are indexed 0 .. (N_theta*N_zeta - 1)
        # Row-major: idx = theta * N_zeta + zeta

        edges = []
        for t in range(self.n_theta):
            for z in range(self.n_zeta):
                curr = t * self.n_zeta + z

                # Neighbors (Up, Down, Left, Right) with periodic BC

                # Theta neighbors (Up/Down)
                t_next = (t + 1) % self.n_theta
                t_prev = (t - 1 + self.n_theta) % self.n_theta

                # Zeta neighbors (Left/Right)
                z_next = (z + 1) % self.n_zeta
                z_prev = (z - 1 + self.n_zeta) % self.n_zeta

                neighbors = [
                    t_next * self.n_zeta + z,
                    t_prev * self.n_zeta + z,
                    t * self.n_zeta + z_next,
                    t * self.n_zeta + z_prev,
                ]

                for nbr in neighbors:
                    edges.append([curr, nbr])

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return edge_index


class GeometricGNN(nn.Module):
    """Graph Neural Network for Stellarator Geometry.

    Replaces PointNet. Operates on the mesh defined by the Fourier Grid.
    Input: (Batch, 3, N_points) or (Batch, 3, H, W)
    Output: Global embedding vector.
    """

    def __init__(
        self,
        n_theta: int,
        n_zeta: int,
        input_dim: int = 3,
        hidden_dim: int = 64,
        output_dim: int = 128,
        num_layers: int = 3,
        align_input: bool = True,
    ):
        super().__init__()
        self.n_theta = n_theta
        self.n_zeta = n_zeta
        self.align_input = align_input

        self.graph_structure = ToroidalGridGraph(n_theta, n_zeta)

        if align_input:
            self.input_transform = equivariance.TNet3d(input_dim=input_dim)

        self.embedding = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(MessagePassingLayer(hidden_dim, hidden_dim))

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Point cloud (Batch, 3, N_points)
               N_points must equal n_theta * n_zeta.
        Returns:
            emb: (Batch, output_dim)
        """
        if self.align_input:
            # T-Net expects (B, 3, N) and returns (B, 3, 3)
            transform = self.input_transform(x)
            # Apply transform: (B, 3, 3) x (B, 3, N) -> (B, 3, N)
            x = torch.bmm(transform, x)

        batch_size, c, n = x.shape

        # Transpose to (Batch, N, C) for Linear layers
        x_node = x.transpose(1, 2)

        # Initial embedding
        h = self.embedding(x_node)  # (B, N, hidden)

        # Message Passing
        edge_index = self.graph_structure.edge_index

        for layer in self.layers:
            h = layer(h, edge_index)

        # Global Pooling
        # h: (B, N, hidden) -> transpose to (B, hidden, N) for pooling
        h_pool = h.transpose(1, 2)
        h_global = self.global_pool(h_pool).squeeze(-1)  # (B, hidden)

        out = self.head(h_global)
        return out
