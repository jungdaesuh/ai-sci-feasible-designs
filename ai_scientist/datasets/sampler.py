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
            # HuggingFace dataset uses flattened keys like "metrics.aspect_ratio"
            ar = ex.get("metrics.aspect_ratio")
            tri = ex.get("metrics.average_triangularity")
            iota = ex.get("metrics.edge_rotational_transform_over_n_field_periods")
            if ar is None or tri is None or iota is None:
                return False
            return abs(ar - 4.0) < 0.1 and abs(tri + 0.6) < 0.1 and iota >= 0.3

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
            # Apply filters only if the example has the corresponding metrics values.
            # The dataset schema is flattened: "metrics.<name>".
            ar = ex.get("metrics.aspect_ratio")
            if ar is not None and ar > 10.0:
                return False

            iota = ex.get("metrics.edge_rotational_transform_over_n_field_periods")
            if iota is not None and iota < 0.25:
                return False

            elong = ex.get("metrics.max_elongation")
            if elong is not None and elong > 5.0:
                return False

            mirror = ex.get("metrics.edge_magnetic_mirror_ratio")
            if mirror is not None and mirror > 0.2:
                return False

            return True

        ds = ds.filter(p2_filter)

    return ds
