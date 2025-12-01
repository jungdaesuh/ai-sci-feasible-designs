import jax
import jax.numpy as jnp
import numpy as np
import pydantic
import pytest
from typing import Any
from ai_scientist.utils.pytree import register_pydantic_data, mask_and_ravel

@register_pydantic_data
class SimpleModel(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
    x: float
    y: jnp.ndarray
    name: str = "test"

def test_pydantic_pytree_roundtrip():
    model = SimpleModel(x=1.0, y=jnp.array([2.0, 3.0]))
    leaves, treedef = jax.tree_util.tree_flatten(model)
    
    assert len(leaves) == 3 # x, y, name? No, name is string so it depends on if it's considered a leaf.
    # Strings are usually leaves in JAX unless filtered?
    # pydantic_flatten treats all non-meta fields as children.
    
    restored = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(restored, SimpleModel)
    assert restored.x == 1.0
    assert jnp.allclose(restored.y, jnp.array([2.0, 3.0]))
    assert restored.name == "test"

def test_mask_and_ravel():
    model = SimpleModel(x=10.0, y=jnp.array([20.0, 30.0]))
    
    # Mask only 'y' for optimization
    # Note: mask structure must match pytree structure
    # Ideally we construct mask using the same model structure
    mask = SimpleModel(x=0.0, y=jnp.array([True, False]), name="ignored") # x=0.0 -> False, y=[T, F]
    # Note: strings in mask usually cause issues if strict checking, but mask_and_ravel handles floats/bools.
    # Our implementation handles float masks (0.0/1.0).
    # But 'name' is a string. mask_and_ravel iterates leaves.
    # If 'name' is a leaf in 'model', it must have a corresponding leaf in 'mask'.
    # 'name'="ignored" is a string.
    
    # We need to handle non-numeric leaves in mask if they are not being masked.
    # Our mask_and_ravel implementation raises ValueError for unsupported mask types.
    # It supports bool array, float (0.0/1.0), bool True/False.
    # So string leaf in mask will fail.
    
    # Solution: Mark 'name' as meta_field in SimpleModel or handle it.
    # Let's define a model where 'name' is metadata.

    @register_pydantic_data
    class NumericModel(pydantic.BaseModel):
        model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
        a: jnp.ndarray
        b: jnp.ndarray

    num_model = NumericModel(a=jnp.array([1.0, 2.0]), b=jnp.array([3.0, 4.0]))
    num_mask = NumericModel(a=jnp.array([True, False]), b=jnp.array([False, True]))
    
    flat, unravel_fn = mask_and_ravel(num_model, num_mask)
    
    # Should contain a[0] (1.0) and b[1] (4.0)
    assert flat.shape == (2,)
    assert flat[0] == 1.0
    assert flat[1] == 4.0
    
    # Modify
    flat_new = jnp.array([10.0, 40.0])
    new_model = unravel_fn(flat_new)
    
    assert new_model.a[0] == 10.0
    assert new_model.a[1] == 2.0 # Unchanged
    assert new_model.b[0] == 3.0 # Unchanged
    assert new_model.b[1] == 40.0

def test_jit_compatibility():
    @jax.jit
    def scale_model(m: SimpleModel) -> SimpleModel:
        # JAX JIT works on leaves.
        # x is float (scalar), y is array.
        # name is string, so it must be treated as metadata (aux_data) to avoid JIT errors.
        # However, SimpleModel didn't register 'name' as a meta_field in the decorator above.
        # By default, pydantic_flatten treats all fields as children unless specified in meta_fields.
        # String children are not valid JAX types.
        
        # To make this work, we need a model where non-array fields are explicitly meta_fields,
        # OR we rely on the fact that we might not be able to JIT models with string children unless
        # we register them carefully.
        
        # Let's use a numeric-only model for JIT testing to be safe, or define a new one with meta_fields.
        return m
    
    @register_pydantic_data
    class JitModel(pydantic.BaseModel):
        model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
        val: jnp.ndarray
        factor: Any

        
    @jax.jit
    def compute(m: JitModel) -> JitModel:
        return JitModel(val=m.val * m.factor, factor=m.factor)
        
    m = JitModel(val=jnp.array([1.0, 2.0]), factor=2.0)
    m_out = compute(m)
    
    assert jnp.allclose(m_out.val, jnp.array([2.0, 4.0]))
    assert m_out.factor == 2.0