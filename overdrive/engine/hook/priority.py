from enum import IntEnum
from typing import Union

__all__ = ["HookPriority"]


class HookPriority(IntEnum):
    VERY_HIGH = 0
    HIGH = 25
    NORMAL = 50
    LOW = 75
    VERY_LOW = 100

    @staticmethod
    def get_priority(priority: Union[int, str, "HookPriority"]) -> int:
        """
        Get priority value with flexible input handling.

        Args:
            priority (int or str or :obj:`HookPriority`): Priority level.
                - int: Direct priority value (0-100)
                - str: Priority name (case-insensitive, e.g., 'high', 'NORMAL')
                - HookPriority: HookPriority enum member

        Returns:
            int: The priority value (0-100).

        Raises:
            ValueError: If priority is invalid or out of range.
            KeyError: If string priority name is not recognized.

        Examples:

            >>> HookPriority.get_priority(50)
            50
            >>> HookPriority.get_priority('HIGH')
            25
            >>> HookPriority.get_priority(HookPriority.LOW)
            75
        """
        if isinstance(priority, HookPriority):
            return priority.value
        elif isinstance(priority, int):
            if not (0 <= priority <= 100):
                raise ValueError(f"Priority must be between 0 and 100, got {priority}. Valid range: [0, 100] where 0=VERY_HIGH, 100=VERY_LOW")
            return priority
        elif isinstance(priority, str):
            priority_str = priority.upper()
            try:
                return HookPriority[priority_str].value
            except KeyError as e:
                valid_names = ", ".join(p.name for p in HookPriority)
                raise KeyError(f"Unknown priority name: '{priority}'. Valid options: {valid_names}") from e
        else:
            raise TypeError(f"Priority must be an int, str, or HookPriority enum, got {type(priority).__name__}")
