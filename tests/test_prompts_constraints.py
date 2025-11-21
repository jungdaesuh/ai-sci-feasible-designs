from ai_scientist import prompts
from constellaration import problems


def _constraint_value(spec: prompts.ProblemSpec, name: str) -> float:
    for constraint in spec.constraints:
        if constraint.name == name:
            return constraint.value
    raise AssertionError(f"constraint {name} missing")


def test_problem_specs_match_problem_definitions():
    geom = problems.GeometricalProblem()
    simple_qi = problems.SimpleToBuildQIStellarator()
    mhd_qi = problems.MHDStableQIStellarator()

    p1_spec = prompts.get_problem_spec("p1")
    assert _constraint_value(p1_spec, "aspect_ratio") == geom._aspect_ratio_upper_bound
    assert (
        _constraint_value(p1_spec, "edge_rotational_transform_over_n_field_periods")
        == geom._edge_rotational_transform_over_n_field_periods_lower_bound
    )
    assert p1_spec.objectives[0].direction == "min"

    p2_spec = prompts.get_problem_spec("p2")
    assert (
        _constraint_value(p2_spec, "aspect_ratio")
        == simple_qi._aspect_ratio_upper_bound
    )
    assert (
        _constraint_value(p2_spec, "edge_magnetic_mirror_ratio")
        == simple_qi._edge_magnetic_mirror_ratio_upper_bound
    )
    assert p2_spec.objectives[0].direction == "max"

    p3_spec = prompts.get_problem_spec("p3")
    assert (
        _constraint_value(p3_spec, "flux_compression_in_regions_of_bad_curvature")
        == mhd_qi._flux_compression_in_regions_of_bad_curvature_upper_bound
    )
    assert _constraint_value(p3_spec, "vacuum_well") == mhd_qi._vacuum_well_lower_bound
    assert {obj.direction for obj in p3_spec.objectives} == {"max", "min"}
