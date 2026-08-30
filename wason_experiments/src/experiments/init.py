# src/experiments/__init__.py
from .baseline import BaselineExperiment
from .dual_goal import DualGoalExperiment
from .think_opposites import ThinkOppositesExperiment

EXPERIMENTS = {
    "baseline": BaselineExperiment,
    "dual-goal": DualGoalExperiment,
    "think-in-opposites": ThinkOppositesExperiment,
}

__all__ = ["BaselineExperiment", "DualGoalExperiment", "ThinkOppositesExperiment", "EXPERIMENTS"]
