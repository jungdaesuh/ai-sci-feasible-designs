import jax
import jax.numpy as jnp
import numpy as np
import pydantic
import pytest
from typing import Any
from pydantic import ConfigDict
from ai_scientist.utils.pytree import register_pydantic_data, mask_and_ravel

@register_pydantic_data
class MyModel(pydantic.BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    a: int
    b: float
    arr: Any = None

def test_pydantic_pytree_registration():
    model = MyModel(a=1, b=2.0, arr=jnp.array([1.0, 2.0]))
    
    # Test tree_flatten
    leaves, treedef = jax.tree_util.tree_flatten(model)
    assert len(leaves) == 3
    assert leaves[0] == 1
    assert leaves[1] == 2.0
    assert jnp.allclose(leaves[2], jnp.array([1.0, 2.0]))
    
    # Test tree_unflatten
    model2 = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(model2, MyModel)
    assert model2.a == 1
    assert model2.b == 2.0
    assert jnp.allclose(model2.arr, jnp.array([1.0, 2.0]))
    
    # Test with JIT
    @jax.jit
    def f(m):
        return m.b * m.arr
        
    res = f(model)
    assert jnp.allclose(res, jnp.array([2.0, 4.0]))

def test_mask_and_ravel():
    model = MyModel(a=1, b=2.0, arr=jnp.array([10.0, 20.0, 30.0]))
    
    # Create a mask structure matching the model
    # Mask 'b' (scalar) and parts of 'arr'
    mask = MyModel(
        a=0, # or False
        b=1.0, # or True
        arr=jnp.array([True, False, True])
    )
    
    flat, unravel_fn = mask_and_ravel(model, mask)
    
    # We expect: b (1 val) + arr[0] + arr[2] = 1 + 1 + 1 = 3 values
    assert flat.shape == (3,)
    # flat should contain [2.0, 10.0, 30.0] (order depends on flattening order, usually defined by field order)
    # MyModel fields are a, b, arr.
    # a is masked out.
    # b is included -> 2.0
    # arr is included at 0 and 2 -> 10.0, 30.0
    expected = jnp.array([2.0, 10.0, 30.0])
    assert jnp.allclose(flat, expected)
    
    # Test unravel
    # Change values
    new_flat = jnp.array([5.0, 100.0, 300.0])
    new_model = unravel_fn(new_flat)
    
    assert isinstance(new_model, MyModel)
    # 'a' should be unchanged
    assert new_model.a == 1
    # 'b' should be updated
    assert new_model.b == 5.0
    # 'arr' should be updated at masked indices, unchanged at unmasked
    expected_arr = jnp.array([100.0, 20.0, 300.0])
    assert jnp.allclose(new_model.arr, expected_arr)

def test_mask_and_ravel_scalar_types():
    # Verify 0/1 integers work as masks for scalars
    val = 10.0
    mask = 1
    flat, unravel = mask_and_ravel(val, mask)
    assert flat[0] == 10.0
    assert unravel(jnp.array([20.0])) == 20.0
    
    mask = 0
    flat, unravel = mask_and_ravel(val, mask)
    assert len(flat) == 0
    assert unravel(jnp.array([])) == 10.0
