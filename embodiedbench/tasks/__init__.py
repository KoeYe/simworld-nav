"""Task plugins (PLAN.md 8, Contract C)."""

from embodiedbench.tasks.core import TaskPlugin, TaskRequirements
from embodiedbench.tasks.delivery import DeliveryTask

__all__ = ["DeliveryTask", "TaskPlugin", "TaskRequirements"]
