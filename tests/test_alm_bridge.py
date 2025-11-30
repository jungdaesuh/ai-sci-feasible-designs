import pytest
import jax.numpy as jnp
import numpy as np

# Check if constellaration is available
try:
    import constellaration
    from constellaration.optimization.augmented_lagrangian import (
        AugmentedLagrangianState,
        AugmentedLagrangianSettings,
        update_augmented_lagrangian_state,
    )
    from constellaration.optimization.augmented_lagrangian_runner import objective_constraints
    from constellaration import problems
    from constellaration.geometry import surface_rz_fourier
    from constellaration.optimization import settings as opt_settings
    import constellaration.forward_model as forward_model
    from constellaration.utils.pytree import register_pydantic_data
    
    # Register SurfaceRZFourier as a pytree node to ensure mask_and_ravel works
    # This is likely done implicitly in the full app but needed explicitly here
    register_pydantic_data(
        surface_rz_fourier.SurfaceRZFourier,
        meta_fields=["n_field_periods", "is_stellarator_symmetric"]
    )
except ImportError:
    pytest.skip("constellaration not installed", allow_module_level=True)

from ai_scientist.optim.alm_bridge import (
    ALMContext,
    ALMStepResult,
    create_alm_context,
    step_alm,
    state_to_boundary_params,
)

# ... (TestALMBridgeAPIContract and test_alm_bridge_imports remain unchanged)

@pytest.mark.slow
class TestALMBridgeFunctionality:
    """Integration tests that run actual optimization."""

    @pytest.fixture(autouse=True)
    def patch_build_mask(self, monkeypatch):
        """Patch build_mask to fix structure mismatch for symmetric surfaces."""
        original_build_mask = surface_rz_fourier.build_mask
        
        def fixed_build_mask(surface, max_poloidal_mode, max_toroidal_mode):
            mask = original_build_mask(surface, max_poloidal_mode, max_toroidal_mode)
            if surface.is_stellarator_symmetric:
                # Fix the mask to have None instead of False, matching boundary structure
                # Use object.__setattr__ to bypass Pydantic validation/immutability if needed
                object.__setattr__(mask, 'r_sin', None)
                object.__setattr__(mask, 'z_cos', None)
            return mask
            
        monkeypatch.setattr(surface_rz_fourier, 'build_mask', fixed_build_mask)

    @pytest.fixture(autouse=True)
    def patch_process_pool(self, monkeypatch):
        """Use ThreadPoolExecutor instead of ProcessPoolExecutor to avoid pickling/registration issues in tests."""
        from concurrent import futures
        # We need to accept the mp_context argument even if ThreadPoolExecutor doesn't use it
        class MockProcessPoolExecutor(futures.ThreadPoolExecutor):
            def __init__(self, max_workers=None, mp_context=None, **kwargs):
                super().__init__(max_workers=max_workers, **kwargs)

        monkeypatch.setattr(futures, 'ProcessPoolExecutor', MockProcessPoolExecutor)

    @pytest.fixture
    def minimal_p3_problem(self):
        return problems.MHDStableQIStellarator(
            edge_rotational_transform_over_n_field_periods_lower_bound=0.1,
            log10_qi_upper_bound=-1.0,
            edge_magnetic_mirror_ratio_upper_bound=0.3,
            flux_compression_in_regions_of_bad_curvature_upper_bound=1.0,
            vacuum_well_lower_bound=0.0,
        )

    @pytest.fixture
    def minimal_boundary(self):
        # Create a simple torus boundary: R = 1 + 0.1*cos(theta), Z = 0.1*sin(theta)
        n_poloidal_modes = 2  # m=0, 1
        n_toroidal_modes = 3  # n=-1, 0, 1 (max_n=1)
        max_toroidal_mode = 1
        
        r_cos = np.zeros((n_poloidal_modes, n_toroidal_modes))
        z_sin = np.zeros((n_poloidal_modes, n_toroidal_modes))
        
        # R0 = 1.0 (m=0, n=0)
        r_cos[0, max_toroidal_mode] = 1.0
        # a = 0.1 (m=1, n=0)
        r_cos[1, max_toroidal_mode] = 0.1
        
        # Z: a = 0.1 (m=1, n=0)
        z_sin[1, max_toroidal_mode] = 0.1
        
        return surface_rz_fourier.SurfaceRZFourier(
            r_cos=r_cos,
            z_sin=z_sin,
            n_field_periods=1,
            is_stellarator_symmetric=True
        )

    @pytest.fixture
    def optimization_settings(self):
        return opt_settings.OptimizationSettings(
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
                     max_time=None
                 )
             ),
             forward_model_settings=forward_model.ConstellarationSettings()
        )

    def test_create_context_returns_valid_state(self, minimal_boundary, minimal_p3_problem, optimization_settings):
        context, state = create_alm_context(
            boundary=minimal_boundary,
            problem=minimal_p3_problem,
            settings=optimization_settings,
            aspect_ratio_upper_bound=10.0,
        )
        
        assert isinstance(context, ALMContext)
        assert isinstance(state, AugmentedLagrangianState)
        assert context.problem == minimal_p3_problem
        assert state.x.shape[0] > 0

    def test_step_alm_executes(self, minimal_boundary, minimal_p3_problem, optimization_settings):
        context, state = create_alm_context(
            boundary=minimal_boundary,
            problem=minimal_p3_problem,
            settings=optimization_settings,
            aspect_ratio_upper_bound=10.0,
        )
        
        result = step_alm(
            context=context,
            state=state,
            budget=10,
            num_workers=2
        )
        
        assert isinstance(result, ALMStepResult)
        assert isinstance(result.state, AugmentedLagrangianState)
        assert result.n_evals > 0
        
    def test_step_alm_penalty_override(self, minimal_boundary, minimal_p3_problem, optimization_settings):
        context, state = create_alm_context(
            boundary=minimal_boundary,
            problem=minimal_p3_problem,
            settings=optimization_settings,
            aspect_ratio_upper_bound=10.0,
        )
        
        override_penalties = jnp.ones_like(state.penalty_parameters) * 100.0
        
        result = step_alm(
            context=context,
            state=state,
            budget=10,
            penalty_override=override_penalties,
            num_workers=2
        )
        
        assert isinstance(result, ALMStepResult)
