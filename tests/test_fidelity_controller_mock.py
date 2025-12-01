
import sys
from unittest.mock import MagicMock, patch

# Mock modules before importing FidelityController
mock_tools = MagicMock()
mock_tools._DEFAULT_RELATIVE_TOLERANCE = 1e-2
mock_tools._P3_REFERENCE_POINT = (1.0, 20.0)
sys.modules["ai_scientist.tools"] = mock_tools

mock_fm = MagicMock()
sys.modules["ai_scientist.forward_model"] = mock_fm

mock_adapter = MagicMock()
sys.modules["ai_scientist.adapter"] = mock_adapter

mock_memory = MagicMock()
sys.modules["ai_scientist.memory"] = mock_memory

mock_runner = MagicMock()
sys.modules["ai_scientist.experiment_runner"] = mock_runner

import unittest
from dataclasses import dataclass
from typing import Any, List, Mapping

@dataclass
class MockEvaluationResult:
    metrics: Any
    objective: float
    constraints: List[float]
    constraint_names: List[str]
    feasibility: float
    is_feasible: bool
    cache_hit: bool
    design_hash: str
    evaluation_time_sec: float
    settings: Any
    fidelity: str
    constraints_map: Mapping[str, float]
    error_message: str | None = None

mock_fm.EvaluationResult = MockEvaluationResult
mock_fm.forward_model_batch = MagicMock()

from ai_scientist import config as ai_config
from ai_scientist.fidelity_controller import FidelityController

@dataclass
class MockBudget:
    wall_clock_minutes: float = 10.0
    n_workers: int = 2
    pool_type: str = "thread"

@dataclass
class MockConfig:
    problem: str = "p3"
    stage_gates: Any = None
    governance: Any = None
    cycles: int = 10

@dataclass
class MockMetrics:
    aspect_ratio: float = 2.0
    minimum_normalized_magnetic_gradient_scale_length: float = 1.5
    max_elongation: float = 1.0
    
    def model_dump(self):
        return {
            "aspect_ratio": self.aspect_ratio,
            "minimum_normalized_magnetic_gradient_scale_length": self.minimum_normalized_magnetic_gradient_scale_length,
            "max_elongation": self.max_elongation
        }

class TestFidelityControllerMock(unittest.TestCase):
    def setUp(self):
        self.config = MockConfig()
        self.controller = FidelityController(self.config)
        self.budgets = MockBudget()

    @patch("ai_scientist.fidelity_controller._time_exceeded")
    def test_evaluate_stage_calls_batch(self, mock_time_exceeded):
        # Setup
        candidates = [{"params": {"x": 1}, "seed": 1}]
        stage = "screen"
        cycle_start = 0.0
        tool_name = "test_tool"
        
        mock_time_exceeded.return_value = False
        mock_tools._settings_for_stage.return_value = MagicMock()
        
        # Mock batch result
        metrics = MockMetrics()
        
        result = MockEvaluationResult(
            metrics=metrics,
            objective=0.5,
            constraints=[0.1],
            constraint_names=["c1"],
            feasibility=0.1,
            is_feasible=False,
            cache_hit=False,
            design_hash="hash1",
            evaluation_time_sec=1.0,
            settings=MagicMock(),
            fidelity="low",
            constraints_map={"c1": 0.1}
        )
        result.settings.constellaration_settings.model_dump.return_value = {}
        
        mock_fm.forward_model_batch.return_value = [result]

        # Execute
        results = self.controller.evaluate_stage(
            candidates,
            stage,
            self.budgets,
            cycle_start,
            evaluate_fn=None,
            tool_name=tool_name
        )

        # Verify
        mock_tools._settings_for_stage.assert_called_with(stage, "p3")
        mock_adapter.prepare_peft_hook.assert_called_with(tool_name, stage)
        mock_fm.forward_model_batch.assert_called_once_with(
            [candidates[0]["params"]],
            mock_tools._settings_for_stage.return_value,
            n_workers=self.budgets.n_workers,
            pool_type=self.budgets.pool_type,
            use_cache=True
        )
        mock_adapter.apply_lora_updates.assert_called_with(tool_name, stage)
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["design_hash"], "hash1")
        self.assertEqual(results[0]["evaluation"]["objective"], 0.5)
        self.assertEqual(results[0]["evaluation"]["score"], 0.75)

    @patch("ai_scientist.fidelity_controller._time_exceeded")
    def test_evaluate_stage_handles_errors(self, mock_time_exceeded):
        # Setup
        candidates = [{"params": {"x": 1}, "seed": 1}]
        stage = "screen"
        cycle_start = 0.0
        tool_name = "test_tool"
        
        mock_time_exceeded.return_value = False
        mock_tools._settings_for_stage.return_value = MagicMock()
        
        # Mock failed batch result
        result = MockEvaluationResult(
            metrics=MockMetrics(aspect_ratio=float("inf"), minimum_normalized_magnetic_gradient_scale_length=0.0, max_elongation=float("inf")),
            objective=1e9,
            constraints=[],
            constraint_names=[],
            feasibility=float("inf"),
            is_feasible=False,
            cache_hit=False,
            design_hash="error",
            evaluation_time_sec=0.0,
            settings=MagicMock(),
            fidelity="low",
            constraints_map={},
            error_message="Simulation failed"
        )
        result.settings.constellaration_settings.model_dump.return_value = {}
        
        mock_fm.forward_model_batch.return_value = [result]

        # Execute
        results = self.controller.evaluate_stage(
            candidates,
            stage,
            self.budgets,
            cycle_start,
            evaluate_fn=None,
            tool_name=tool_name
        )

        # Verify
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["evaluation"]["error"], "Simulation failed")
        self.assertEqual(results[0]["evaluation"]["feasibility"], float("inf"))
