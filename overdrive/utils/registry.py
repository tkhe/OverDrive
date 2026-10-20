import copy
from collections.abc import Callable
from collections.abc import ItemsView
from collections.abc import Iterator
from collections.abc import KeysView
from collections.abc import ValuesView
from typing import Any

from tabulate import tabulate

__all__ = ["Registry"]


class Registry:
    """
    The registry that provides name -> object mapping, to support third-party
    user's custom modules.

    Registered object could be built from registry. An optional ``alias`` can be
    provided to register the same object under one or more alternative names.

    Examples:

        >>> from overdrive.utils.registry import Registry
        >>> # create a registry
        >>> BACKBONE_REGISTRY = Registry("backbone")
        >>> # to register a module, used as a decorator
        >>> @BACKBONE_REGISTRY.register()
        ... class MyBackbone:
        ...     pass
        >>> # to register a module with an alias
        >>> @BACKBONE_REGISTRY.register(alias="mb")
        ... class MyBackbone2:
        ...     pass
        >>> # to register a module with multiple aliases
        >>> @BACKBONE_REGISTRY.register(alias=["res50", "r50"])
        ... class ResNet50:
        ...     pass
        >>> # to register a module, used as a function call
        >>> BACKBONE_REGISTRY.register("AnyBackbone", MyBackbone)
        >>> # to register a module with an alias, used as a function call
        >>> BACKBONE_REGISTRY.register("AnyBackbone2", MyBackbone2, alias="ab2")
        >>> # build object from registry (by name or alias)
        >>> obj = BACKBONE_REGISTRY.build(dict(name="AnyBackbone"))
        >>> obj = BACKBONE_REGISTRY.build(dict(name="ab2"))
    """

    def __init__(self, name: str, allow_override: bool = False):
        """
        Args:
            name (str): the name of registry.
            allow_override (bool): If True, the previously registered object will
                be overriden when the same name is registered.
        """
        self._name = name
        self._allow_override = allow_override

        self._obj_map: dict[str, Any] = {}

    def _do_register(self, name: str, obj: Any):
        if name in self._obj_map and not self._allow_override:
            raise ValueError(f"An object named '{name}' was already registered in '{self._name}' registry!")

        self._obj_map[name] = obj

    def register(
        self,
        name: str | None = None,
        obj: Any | None = None,
        alias: str | list[str] | None = None,
    ) -> Callable[[Any], Any] | None:
        """
        Register the given object under the the name or `obj.__name__` if name is None.
        Can be used as either a decorator or not. See docstring of this class for usage.

        Args:
            name (str, optional): the name to register the object under. If None,
                `obj.__name__` will be used.
            obj (Any, optional): the object to register. If None, the method acts
                as a decorator.
            alias (str or list[str], optional): one or more aliases for the object.
                Each alias is registered as an independent key pointing to the same object.
        """
        # normalize alias to a list
        if alias is None:
            alias_list: list[str] = []
        elif isinstance(alias, str):
            alias_list = [alias]
        else:
            alias_list = list(alias)

        if obj is None:
            # used as a decorator
            def deco(func_or_class: Any) -> Any:
                nonlocal name

                if name is None:
                    name = func_or_class.__name__

                self._do_register(name, func_or_class)
                for a in alias_list:
                    self._do_register(a, func_or_class)
                return func_or_class

            return deco

        # used as a function call
        if name is None:
            name = obj.__name__

        self._do_register(name, obj)
        for a in alias_list:
            self._do_register(a, obj)

    def keys(self) -> KeysView[str]:
        return self._obj_map.keys()

    def values(self) -> ValuesView[Any]:
        return self._obj_map.values()

    def items(self) -> ItemsView[str, Any]:
        return self._obj_map.items()

    def get(self, name: str) -> Any:
        ret = self._obj_map.get(name)

        if ret is None:
            raise KeyError(f"No object named '{name}' found in '{self._name}' registry!")

        return ret

    def build(self, cfg: dict[str, Any], **extra_kwargs: Any) -> Any:
        """
        Build an instance.

        Args:
            cfg (dict): config dict need to be built.

        Return:
            Any: The constructed object.
        """
        kwargs = copy.deepcopy(cfg)
        kwargs.update(**extra_kwargs)

        if "name" not in kwargs:
            raise KeyError("`cfg` or `extra_kwargs` must contain the key `name`")

        name = kwargs.pop("name")
        obj = self.get(name)

        try:
            return obj(**kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to build object '{name}' from registry '{self._name}' with error: {e}") from e

    def __contains__(self, name: str) -> bool:
        return name in self._obj_map

    def __len__(self) -> int:
        return len(self._obj_map)

    def __repr__(self) -> str:
        table_headers = ["Name", "Objects"]
        table = tabulate(self._obj_map.items(), headers=table_headers, tablefmt="fancy_grid")
        return f"Registry of {self._name}\n{table}"

    def __iter__(self) -> Iterator[tuple[str, Any]]:
        return iter(self._obj_map.items())

    def __str__(self) -> str:
        return self.__repr__()
