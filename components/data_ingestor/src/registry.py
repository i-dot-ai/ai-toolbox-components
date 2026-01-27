"""
Generic plugin registry with auto-discovery.

Provides a reusable mechanism for discovering and registering plugin classes
(e.g., parsers, embedders) from a package by scanning for subclasses of a
given base class.
"""

import importlib
import logging
import pkgutil
from pathlib import Path

logger = logging.getLogger(__name__)


class PluginRegistry:
    """
    Auto-discovers and registers plugin classes from a Python package.

    Scans all modules in a package for subclasses of a given base class
    and registers them by a configurable key attribute (e.g., source_type,
    store_type).

    Args:
        base_class: The abstract base class that plugins must inherit from.
        key_attr: The property name used as the registry key.
        label: Human-readable label for log messages (e.g., "parser", "embedder").
    """

    def __init__(self, base_class: type, key_attr: str, label: str) -> None:
        self._base_class = base_class
        self._key_attr = key_attr
        self._label = label
        self._registry: dict[str, type] = {}

    def discover(self, package_name: str, package_dir: Path) -> None:
        """
        Scan a package directory and register all discovered plugin classes.

        Args:
            package_name: Fully qualified package name for imports.
            package_dir: Filesystem path to the package directory.
        """
        for module_info in pkgutil.iter_modules([str(package_dir)]):
            if module_info.name == "base":
                continue
            try:
                module = importlib.import_module(f".{module_info.name}", package=package_name)
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (
                        isinstance(attr, type)
                        and issubclass(attr, self._base_class)
                        and attr is not self._base_class
                    ):
                        instance = attr.__new__(attr)
                        key = getattr(instance, self._key_attr)
                        self._registry[key] = attr
                        logger.info(f"Registered {self._label}: {key} -> {attr.__name__}")
            except Exception as e:
                logger.warning(f"Failed to load {self._label} module '{module_info.name}': {e}")

        logger.info(
            f"{self._label.title()} discovery complete: {len(self._registry)} registered "
            f"({list(self._registry.keys())})"
        )

    def get(self, key: str) -> type:
        """
        Look up a plugin class by its key.

        Args:
            key: The registry key (e.g. 'html' for a parser).

        Returns:
            The plugin class.

        Raises:
            ValueError: If the key is not found in the registry.
        """
        if key not in self._registry:
            raise ValueError(
                f"Unknown {self._label}: {key}. "
                f"Available: {list(self._registry.keys())}"
            )
        return self._registry[key]

    def supported_keys(self) -> list[str]:
        """Return list of registered keys."""
        return list(self._registry.keys())
