import dataclasses
from collections.abc import Iterable
from typing import Any, Callable, TypeVar, Union, List, Tuple, Dict

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import pydantic

NpOrJaxArray = Union[np.ndarray, jnp.ndarray]


def pydantic_flatten(
    something: pydantic.BaseModel,
    meta_fields: List[str] | None = None,
) -> Tuple[
    Tuple[Tuple[jtu.GetAttrKey, Any], ...],
    Tuple[type[pydantic.BaseModel], Tuple[str, ...], Tuple[str, ...], Tuple[Any, ...]],
]:
    """A jax pytree compatible implementation of flattening pydantic objects.

    A general pydantic.BaseModel is flattened into a tuple of children and aux_data.
    aux_data is used to reconstruct the same type of Pydantic in unflatten. meta_fields
    are used to specify fields that should not be visible to pytree operations. See
    https://jax.readthedocs.io/en/latest/_autosummary/jax.tree_util.register_dataclass.html
    for details.
    """

    if meta_fields is None:
        meta_fields = []

    data_fields = [
        field for field in type(something).model_fields if field not in meta_fields
    ]

    aux_data = (
        type(something),
        tuple(data_fields),
        tuple(meta_fields),
        tuple(getattr(something, field) for field in meta_fields),
    )
    children = tuple(
        (jtu.GetAttrKey(field), getattr(something, field)) for field in data_fields
    )
    return children, aux_data


def pydantic_unflatten(
    aux_data: Tuple[
        type[pydantic.BaseModel], Tuple[str, ...], Tuple[str, ...], Tuple[Any, ...]
    ],
    children: Iterable[Any],
) -> pydantic.BaseModel:
    """A jax pytree compatible implementation of un-flattening Pydantic objects.

    pydantic_unflatten is the inverse of pydantic_flatten. Using the type annotation
    and children-meta_fields split in aux_data, it constructs a new Pydantic object.

    Note that this object will typically not validate against the Pydantic
    specification. Pytrees are used to manipulate the data stored in leafs and thus can
    contain anything. Pytrees only make statements about the structure of the data, not
    the content. An example of data manipulation might be to filter data in a pytree by
    type:

    ```
    my_data = MyModel(...)
    strings = jax.tree_map(lambda x: x if isinstance(x, str) else None, my_data)
    ```

    See
    https://jax.readthedocs.io/en/latest/pytrees.html#custom-pytrees-and-initialization
    for details.
    """
    cls, data_fields, meta_fields, metas = aux_data
    kwargs = {
        **dict(zip(data_fields, children)),
        **dict(zip(meta_fields, metas)),
    }
    # Bypass validation for speed and flexibility during JAX transformations
    return cls.model_construct(**kwargs)


def register_pydantic_data(cls: type, meta_fields: List[str] | None = None) -> type:
    """Register a pydantic.BaseModel class for jax pytree compatibility.

    Args:
        cls: The pydantic.BaseModel class to register.
        meta_fields: Fields that should be part of aux_data rather becoming children.
            Defaults to None.

    Returns:
        The registered class, to enable the function to be used as a decorator.
    """
    jtu.register_pytree_with_keys(
        cls,
        lambda x: pydantic_flatten(x, meta_fields),
        pydantic_unflatten,
    )

    return cls


PytreeT = TypeVar("PytreeT")
LeafT = TypeVar("LeafT")


@dataclasses.dataclass
class _LeafInfo:
    leaf: Union[NpOrJaxArray, float]
    mask: Union[NpOrJaxArray, None]
    is_scalar: bool

    @property
    def is_masked(self) -> bool:
        return self.mask is not None


class _UnravelFn:
    """A picklable callable that reconstructs a pytree from a flat parameter vector.

    It stores all necessary info (the original pytree treedef and a list of per-leaf
    info) to reassemble the structure.
    """

    _leaves_info: List[_LeafInfo]
    _treedef: jax.tree_util.PyTreeDef

    def __init__(
        self,
        leaves_info: List[_LeafInfo],
        treedef: jax.tree_util.PyTreeDef,
    ) -> None:
        self._leaves_info = leaves_info
        self._treedef = treedef

    def __call__(self, flat: NpOrJaxArray) -> Any:
        new_leaves: List[Union[NpOrJaxArray, float]] = []
        pos = 0
        for leaf_info in self._leaves_info:
            if leaf_info.is_masked:
                assert leaf_info.mask is not None
                # Determine how many entries were extracted for this leaf.
                n_true = int(leaf_info.mask.sum())
                segment = flat[pos : pos + n_true]
                pos += n_true
                # Find the indices where the mask is True.
                idx = jnp.nonzero(leaf_info.mask)
                if leaf_info.is_scalar:
                    # If the leaf is a scalar, `segment` has a single entry.
                    new_leaf = segment[0].item()
                else:
                    # Replace the masked positions with the new parameters.
                    # Note: we assume leaf is array-like if not scalar
                    new_leaf = jnp.asarray(leaf_info.leaf).at[idx].set(segment)
                new_leaves.append(new_leaf)
            else:
                new_leaves.append(leaf_info.leaf)
        if pos != len(flat):
            raise ValueError(
                f"Expected to consume {len(flat)} values, but only consumed {pos}."
            )
        return jax.tree_util.tree_unflatten(self._treedef, new_leaves)


def mask_and_ravel(
    pytree: PytreeT,
    mask: PytreeT,
) -> Tuple[NpOrJaxArray, Callable[[NpOrJaxArray], PytreeT]]:
    """Ravel a pytree but only include entries where the corresponding mask is True.
    Returns a 1D array and a picklable unravel function that reconstructs the pytree
    by inserting new values into the masked locations while keeping other
    entries unchanged.

    Args:
        pytree: The pytree to be flattened.
        mask: A pytree of booleans with the same structure as `pytree`. True indicates
            the entries to be flattened.

    Returns:
        A tuple of the flat array and the unravel function.
    """

    leaves, treedef = jax.tree_util.tree_flatten(pytree)
    mask_leaves, _ = jax.tree_util.tree_flatten(mask)

    flat_segments: List[NpOrJaxArray] = []
    leaves_info: List[_LeafInfo] = []
    for leaf, leaf_mask in zip(leaves, mask_leaves):
        # Determine if the leaf is effectively a scalar
        # We treat python float/int as scalar, and 0-d arrays as scalar if mask is scalar
        is_leaf_array = hasattr(leaf, "shape") and leaf.shape != ()

        # Check for boolean array mask (explicit array mask)
        if (
            isinstance(leaf_mask, (np.ndarray, jnp.ndarray))
            and leaf_mask.dtype == bool
            and leaf_mask.shape != ()
        ) or (
            hasattr(leaf_mask, "dtype")
            and leaf_mask.dtype == jnp.bool_
            and getattr(leaf_mask, "shape", ()) != ()
        ):
            selected = leaf[leaf_mask]
            flat_segments.append(jnp.atleast_1d(selected.ravel()))
            leaves_info.append(_LeafInfo(mask=leaf_mask, leaf=leaf, is_scalar=False))

        # Scalar masks (int 0/1, float 0.0/1.0, bool True/False, or 0-d array)
        else:
            # Convert JAX scalar to python scalar for checking
            if hasattr(leaf_mask, "item"):
                mask_val = leaf_mask.item()
            else:
                mask_val = leaf_mask

            if mask_val not in (0, 1, 0.0, 1.0, False, True):
                raise ValueError(
                    f"Unsupported mask value {leaf_mask} for leaf {leaf}. "
                    "Scalar masks must be 0/1, False/True."
                )

            is_masked_in = bool(mask_val)
            
            if is_masked_in:
                if is_leaf_array:
                    # Scalar True mask on array leaf -> select all elements
                    # Construct full boolean mask to ensure reconstruction consumes all elements
                    full_mask = jnp.ones_like(leaf, dtype=bool)
                    flat_segments.append(leaf.ravel())
                    leaves_info.append(
                        _LeafInfo(mask=full_mask, leaf=leaf, is_scalar=False)
                    )
                else:
                    # Scalar True mask on scalar leaf
                    flat_segments.append(jnp.atleast_1d(leaf))
                    leaves_info.append(
                        _LeafInfo(
                            mask=jnp.array([True]), leaf=leaf, is_scalar=True
                        )
                    )
            else:
                # Mask is False/0 -> ignore leaf
                leaves_info.append(
                    _LeafInfo(
                        leaf=leaf, mask=None, is_scalar=not is_leaf_array
                    )
                )

    if flat_segments:
        flat = jnp.concatenate(flat_segments)
    else:
        flat = jnp.array([])

    unravel_fn = _UnravelFn(leaves_info, treedef)
    return flat, unravel_fn
