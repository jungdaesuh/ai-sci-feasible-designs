from typing import Optional

from datasets import load_dataset


def load_constellaration_dataset(
    split: str = "train",
    problem: Optional[str] = None,
):
    """
    Load HuggingFace constellaration dataset with optional problem-specific filtering.

    Args:
        split: Dataset split to load (e.g., "train").
        problem: Optional problem identifier ("p1", "p2") to filter the dataset.
                 - p1: Geometrical Problem (aspect_ratio < 4.1, triangularity < -0.5)
                 - p2: Simple QI (aspect_ratio < 10, iota > 0.25, mirror < 0.2, elongation < 5)

    Returns:
        Filtered HuggingFace dataset.
    """
    ds = load_dataset("proxima-fusion/constellaration", split=split)

    if problem == "p1":
        # P1: Geometrical Problem
        # Constraints:
        # 1. aspect_ratio <= 4.0 (using < 4.1 for slight tolerance/inclusive)
        # 2. average_triangularity <= -0.5 (using < -0.49 for tolerance)
        # 3. edge_rotational_transform >= 0.3 (not strictly filtered here to allow some exploration,
        #    but primary geometry constraints are A and delta)
        ds = ds.filter(
            lambda ex: (
                ex["aspect_ratio"] < 4.1 and ex["average_triangularity"] < -0.49
            )
        )
    elif problem == "p2":
        # P2: Simple-to-Build QI Stellarator
        # Constraints:
        # 1. aspect_ratio <= 10.0
        # 2. edge_rotational_transform >= 0.25
        # 3. edge_magnetic_mirror_ratio <= 0.2
        # 4. max_elongation <= 5.0
        ds = ds.filter(
            lambda ex: (
                ex["aspect_ratio"] <= 10.0
                and ex["edge_rotational_transform_over_n_field_periods"] >= 0.25
                and ex["edge_magnetic_mirror_ratio"] <= 0.2
                and ex["max_elongation"] <= 5.0
            )
        )

    return ds
