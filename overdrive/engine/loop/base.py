import bisect
import logging
import weakref

from overdrive.engine.backend import Backend
from overdrive.engine.hook import Hook
from overdrive.utils.events import EventStorage

__all__ = ["Loop"]


class Loop:
    mode: str | None = None

    def __init__(self, backend: Backend):
        self.backend = backend
        self.hooks: list[Hook] = []
        self.storage: EventStorage

    def register_hook(self, hook: Hook | None):
        if hook is not None:
            hook.loop = weakref.proxy(self)

            priorities = [h.priority for h in self.hooks]
            index = bisect.bisect_right(priorities, hook.priority)
            self.hooks.insert(index, hook)

    def register_hooks(self, hooks: list[Hook | None]):
        for hook in hooks:
            self.register_hook(hook)

    def run(self, start: int, max: int):
        raise NotImplementedError

    def run_step(self):
        self.backend.run_step()

    def before_train(self):
        for hook in self.hooks:
            hook.before_train()

    def before_epoch(self):
        for hook in self.hooks:
            hook.before_epoch()

    def before_step(self):
        for hook in self.hooks:
            hook.before_step()

    def after_step(self):
        for hook in self.hooks:
            hook.after_step()

    def after_epoch(self):
        for hook in self.hooks:
            hook.after_epoch()

    def after_train(self):
        for hook in self.hooks:
            hook.after_train()

    def state_dict(self):
        hook_state = {}
        for h in self.hooks:
            sd = h.state_dict()
            if sd:
                name = type(h).__qualname__
                if name in hook_state:
                    # TODO handle repetitive stateful hooks
                    continue
                hook_state[name] = sd
        if hook_state:
            return {"hooks": hook_state}
        else:
            return {}

    def load_state_dict(self, state_dict):
        logger = logging.getLogger(__name__)

        for key, value in state_dict.get("hooks", {}).items():
            for h in self.hooks:
                name = type(h).__qualname__
                if name == key:
                    h.load_state_dict(value)
                    break
            else:
                logger.warning(f"Cannot find the hook '{key}', its state_dict is ignored.")
