import pytest
import torch

from ai_scientist.optim.equivariance import quaternion_to_matrix, random_rotation_matrix


@pytest.mark.parametrize("batch_size", [1, 5, 100])
@pytest.mark.parametrize("device", ["cpu", pytest.param("mps", marks=pytest.mark.mps)])
def test_random_rotation_matrix_properties(batch_size, device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS not available")

    rot_matrices = random_rotation_matrix(batch_size, device=torch.device(device))

    # Test shape
    assert rot_matrices.shape == (batch_size, 3, 3)

    # Test orthogonality (R @ R_T should be identity)
    # Using a small tolerance for floating point comparisons
    identity_batch = (
        torch.eye(3, device=torch.device(device)).unsqueeze(0).repeat(batch_size, 1, 1)
    )

    # Calculate R @ R.T
    rr_t = torch.bmm(rot_matrices, rot_matrices.transpose(-1, -2))
    assert torch.allclose(rr_t, identity_batch, atol=1e-6)

    # Test determinant (+1 for proper rotations)
    # This also implicitly checks that they are not reflections (det=-1)
    det_values = torch.det(rot_matrices)
    assert torch.allclose(
        det_values, torch.ones(batch_size, device=torch.device(device)), atol=1e-6
    )


@pytest.mark.parametrize("device", ["cpu", pytest.param("mps", marks=pytest.mark.mps)])
def test_quaternion_to_matrix_identity(device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS not available")

    # Identity quaternion [1, 0, 0, 0] (w, x, y, z)
    q_identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=torch.device(device))

    mat = quaternion_to_matrix(q_identity)

    expected = torch.eye(3, device=torch.device(device)).unsqueeze(0)

    assert torch.allclose(mat, expected, atol=1e-6)


@pytest.mark.parametrize("device", ["cpu", pytest.param("mps", marks=pytest.mark.mps)])
def test_quaternion_to_matrix_x_rotation(device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS not available")

    # Rotation of 90 degrees around X axis
    # q = [cos(45), sin(45), 0, 0] = [0.7071, 0.7071, 0, 0]
    val = 0.70710678
    q_x_90 = torch.tensor([[val, val, 0.0, 0.0]], device=torch.device(device))

    mat = quaternion_to_matrix(q_x_90)

    # Expected:
    # [[1,  0,  0],
    #  [0,  0, -1],
    #  [0,  1,  0]]
    expected = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        device=torch.device(device),
    ).unsqueeze(0)

    assert torch.allclose(mat, expected, atol=1e-5)
