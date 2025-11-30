import jax
import jax.numpy as jnp
import numpy as np
import pydantic
import pytest
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
        # Just scale numeric fields? JAX JIT works on leaves.
        # x is float, y is array. name is string (might be static or cause tracer error if manipulated).
        # JAX usually treats strings as auxiliary data if registered correctly, OR errors if they are leaves.
        # pydantic_flatten puts everything in children.
        # Strings in children usually break JIT unless they are marked static?
        # No, JAX leaves must be arrays or scalars. Strings are not valid JAX types for computation.
        # They should be in aux_data (meta_fields).
        return SimpleModel(x=m.x * 2.0, y=m.y * 2.0, name=m.name)

    # We need to register 'name' as meta_field for JIT to work if we don't manipulate it.
    pass