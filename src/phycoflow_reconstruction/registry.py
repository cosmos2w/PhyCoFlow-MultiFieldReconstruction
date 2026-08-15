"""Small version-aware registries for models, cases, and future coherence families."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class RegistryEntry(Generic[T]):
    name: str
    version: str
    factory: Callable[..., T]
    metadata: dict[str, Any]


class Registry(Generic[T]):
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._entries: dict[str, RegistryEntry[T]] = {}

    def register(
        self,
        name: str,
        factory: Callable[..., T],
        *,
        version: str = "1",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        key = name.strip().lower()
        if not key:
            raise ValueError(f"{self.kind} name cannot be empty")
        if key in self._entries:
            current = self._entries[key]
            raise KeyError(
                f"duplicate {self.kind} {key!r}; already registered at version {current.version}"
            )
        self._entries[key] = RegistryEntry(key, version, factory, metadata or {})

    def get(self, name: str) -> RegistryEntry[T]:
        key = name.strip().lower()
        if key not in self._entries:
            raise KeyError(f"unknown {self.kind} {name!r}; available={sorted(self._entries)}")
        return self._entries[key]

    def build(self, name: str, **kwargs: Any) -> T:
        return self.get(name).factory(**kwargs)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries))


MODEL_REGISTRY: Registry[Any] = Registry("model")
CASE_REGISTRY: Registry[Any] = Registry("case")
COHERENCE_FAMILY_REGISTRY: Registry[Any] = Registry("coherence family")
