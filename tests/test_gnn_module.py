
import pytest
import torch
from ai_scientist.optim import gnn

def test_toroidal_grid_edges():
    n_theta = 4
    n_zeta = 4
    graph = gnn.ToroidalGridGraph(n_theta, n_zeta)
    
    assert graph.num_nodes == 16
    # Each node has 4 neighbors
    # Total edges = 16 * 4 = 64
    assert graph.edge_index.shape == (2, 64)
    
    # Check a specific connection
    # Node 0 (0,0) should connect to:
    # (1,0) -> 4
    # (3,0) -> 12 (wrap)
    # (0,1) -> 1
    # (0,3) -> 3 (wrap)
    
    neighbors = graph.edge_index[1, graph.edge_index[0] == 0].tolist()
    neighbors.sort()
    assert neighbors == [1, 3, 4, 12]

def test_message_passing_layer():
    layer = gnn.MessagePassingLayer(in_channels=16, out_channels=32)
    
    batch_size = 2
    num_nodes = 10
    x = torch.randn(batch_size, num_nodes, 16)
    
    # Create dummy edges (fully connected for simplicity)
    src = torch.arange(num_nodes).repeat_interleave(num_nodes)
    dst = torch.arange(num_nodes).repeat(num_nodes)
    edge_index = torch.stack([src, dst], dim=0)
    
    out = layer(x, edge_index)
    
    assert out.shape == (batch_size, num_nodes, 32)

def test_geometric_gnn_forward():
    n_theta = 8
    n_zeta = 16
    num_points = n_theta * n_zeta
    
    gnn_model = gnn.GeometricGNN(n_theta, n_zeta, output_dim=64, align_input=False)
    
    batch_size = 3
    # Input shape: (Batch, 3, N_points)
    x = torch.randn(batch_size, 3, num_points)
    
    out = gnn_model(x)
    
    assert out.shape == (batch_size, 64)

def test_geometric_gnn_with_alignment():
    n_theta = 8
    n_zeta = 8
    num_points = n_theta * n_zeta
    
    # Initialize with alignment (T-Net)
    gnn_model = gnn.GeometricGNN(n_theta, n_zeta, output_dim=64, align_input=True)
    
    batch_size = 2
    x = torch.randn(batch_size, 3, num_points)
    
    out = gnn_model(x)
    assert out.shape == (batch_size, 64)
