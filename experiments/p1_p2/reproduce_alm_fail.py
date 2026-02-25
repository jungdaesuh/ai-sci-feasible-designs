
import numpy as np
import jax.numpy as jnp
from constellaration import problems
from constellaration.geometry import surface_rz_fourier
from constellaration.optimization import settings as opt_settings
from constellaration.optimization.augmented_lagrangian import AugmentedLagrangianSettings
import constellaration.forward_model as forward_model
from ai_scientist.optim.alm_bridge import create_alm_context

def reproduce():
    print("Setting up inputs...")
    # Minimal boundary
    r_cos = np.zeros((2, 2))
    z_sin = np.zeros((2, 2))
    r_cos[0, 1] = 1.0
    r_cos[1, 1] = 0.1
    z_sin[1, 1] = 0.1
    
    boundary = surface_rz_fourier.SurfaceRZFourier(
        r_cos=r_cos, z_sin=z_sin, n_field_periods=1, is_stellarator_symmetric=True
    )

    # Minimal problem
    problem = problems.MHDStableQIStellarator(
        edge_rotational_transform_over_n_field_periods_lower_bound=0.1,
        log10_qi_upper_bound=-1.0,
        edge_magnetic_mirror_ratio_upper_bound=0.3,
        flux_compression_in_regions_of_bad_curvature_upper_bound=1.0,
        vacuum_well_lower_bound=0.0,
    )

    # Settings
    settings = opt_settings.OptimizationSettings(
        max_poloidal_mode=2,
        max_toroidal_mode=2,
        infinity_norm_spectrum_scaling=1.0,
        optimizer_settings=opt_settings.AugmentedLagrangianMethodSettings(
            maxit=2,
            penalty_parameters_initial=1.0,
            bounds_initial=0.1,
            augmented_lagrangian_settings=AugmentedLagrangianSettings(),
            oracle_settings=opt_settings.NevergradSettings(
                budget_initial=10,
                budget_increment=10,
                budget_max=100,
                num_workers=2,
                batch_mode=False,
                max_time=None,
            ),
        ),
        forward_model_settings=forward_model.ConstellarationSettings(),
    )

    print("Calling create_alm_context...")
    try:
        context, state = create_alm_context(
            boundary=boundary,
            problem=problem,
            settings=settings,
            aspect_ratio_upper_bound=10.0,
        )
        print("Success!")
    except Exception as e:
        print(f"Caught exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    reproduce()
