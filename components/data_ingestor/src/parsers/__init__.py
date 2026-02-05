"""
Parser package for data ingestion.

Provides auto-discovery of parser classes and a registry for lookup by source type.
"""

from pathlib import Path

from registry import PluginRegistry

from .base import BaseParser, ParsedDocument

_registry = PluginRegistry(BaseParser, "source_type", "parser")
_registry.discover(__name__, Path(__file__).parent)

get_parser_class = _registry.get
supported_types = _registry.supported_keys


__all__ = [
    "BaseParser",
    "ParsedDocument",
    "get_parser_class",
    "supported_types",
]
