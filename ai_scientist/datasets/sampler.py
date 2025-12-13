from typing import Optional
from datasets import load_dataset  # type: ignore


def load_constellaration_dataset(
    split: str = "train",
    problem: Optional[str] = None,
):
    """Load HuggingFace constellaration dataset.

    Args:
        split: Dataset split to load (e.g., "train", "test").
        problem: Optional problem identifier ("p1", "p2", "p3") to filter the dataset.
                 If None, returns the full dataset.

    Returns:
        A HuggingFace Dataset object.
    """
    ds = load_dataset("proxima-fusion/constellaration", split=split)

    if problem == "p1":
        # P1: Geometrical Problem
        # Target: Aspect Ratio ~ 4.0, Triangularity ~ -0.6
        def p1_filter(ex):
            try:
                return (
                    "aspect_ratio" in ex
                    and "average_triangularity" in ex
                    and abs(ex["aspect_ratio"] - 4.0) < 0.1
                    and abs(ex["average_triangularity"] + 0.6) < 0.1
                    and "edge_rotational_transform_over_n_field_periods" in ex
                    and ex["edge_rotational_transform_over_n_field_periods"] >= 0.3
                )
            except (KeyError, TypeError):
                return False

        ds = ds.filter(p1_filter)
    elif problem == "p2":
        # P2: Simple-to-Build QI
        # Filter based on P2 constraints (where columns are available in the dataset):
        # 1. aspect_ratio <= 10.0
        # 2. edge_rotational_transform_over_n_field_periods >= 0.25
        # 3. max_elongation <= 5.0
        # 4. edge_magnetic_mirror_ratio <= 0.2 (if available)
        # Note: qi filtering requires physics evaluation, so we skip it here

        def p2_filter(ex):
            # Apply filters only if the columns exist in the example
            try:
                # Check aspect ratio (if available)
                if "aspect_ratio" in ex:
                    if ex["aspect_ratio"] > 10.0:
                        return False

                # Check edge rotational transform (if available)
                if "edge_rotational_transform_over_n_field_periods" in ex:
                    if ex["edge_rotational_transform_over_n_field_periods"] < 0.25:
                        return False

                # Check max elongation (if available)
                if "max_elongation" in ex:
                    if ex["max_elongation"] > 5.0:
                        return False

                # Check edge magnetic mirror ratio (if available)
                if "edge_magnetic_mirror_ratio" in ex:
                    if ex["edge_magnetic_mirror_ratio"] > 0.2:
                        return False

                return True
            except (KeyError, TypeError):
                # If there's any issue accessing fields, reject the example to avoid corrupted data
                return False

        ds = ds.filter(p2_filter)

    return ds
