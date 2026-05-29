"""Mixin classes for common patterns."""

from __future__ import annotations

from typing import Any


class DictBackedMixin:
    """Mixin providing dictionary-like access to object attributes.

    Classes using this mixin can be accessed like a dictionary:
    - obj["key"] returns obj.key
    - "key" in obj checks hasattr(obj, "key")

    This provides backward compatibility with code that expects
    dictionary-like access to configuration objects.
    """

    def __getitem__(self, key: str) -> Any:
        """Get attribute value by key (dictionary-style access).

        Args:
            key: Attribute name to access.

        Returns:
            Value of the attribute.

        Raises:
            AttributeError: If attribute doesn't exist.
        """
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        """Check if attribute exists (dictionary-style membership test).

        Args:
            key: Attribute name to check.

        Returns:
            True if attribute exists.
        """
        return hasattr(self, key)
