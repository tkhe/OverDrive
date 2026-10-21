from typing import TYPE_CHECKING
from typing import Optional

from .priority import HookPriority

if TYPE_CHECKING:
    from overdrive.engine.loop import Loop

__all__ = ["Hook"]


class Hook:
    loop: Optional["Loop"] = None

    def before_train(self):
        pass

    def before_epoch(self):
        pass

    def before_step(self):
        pass

    def after_step(self):
        pass

    def after_epoch(self):
        pass

    def after_train(self):
        pass

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state_dict: dict):
        pass

    @property
    def priority(self) -> HookPriority:
        if hasattr(self, "_priority"):
            return self._priority
        return HookPriority.NORMAL

    @priority.setter
    def priority(self, value: int | str | HookPriority):
        self._priority = HookPriority.get_priority(value)
