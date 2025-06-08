"""Data models for scenario definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum


class StepType(Enum):
    """Types of scenario steps."""
    WORKLOAD = "workload"
    WAIT = "wait"
    CHECKPOINT = "checkpoint"


@dataclass
class ScenarioStep:
    """Single step in a scenario."""
    name: str
    type: StepType
    workload: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    duration_seconds: Optional[float] = None
    wait_condition: Optional[Dict[str, Any]] = None
    
    def validate(self) -> bool:
        """Validate step configuration."""
        # TODO: Check required fields based on type
        # TODO: Validate parameter values
        raise NotImplementedError()


@dataclass
class Scenario:
    """Complete scenario definition."""
    name: str
    description: str
    dataset: str
    steps: List[ScenarioStep]
    global_config: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> bool:
        """Validate scenario configuration."""
        # TODO: Validate all steps
        # TODO: Check step dependencies
        # TODO: Verify dataset reference
        raise NotImplementedError()
        
    def get_total_duration(self) -> Optional[float]:
        """Calculate total scenario duration if deterministic."""
        # TODO: Sum step durations
        # TODO: Return None if any step has no duration
        raise NotImplementedError()
