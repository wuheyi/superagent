"""Abstractions for interacting with mem0 long-term memory."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

try:
    from mem0 import MemoryClient  # type: ignore
except Exception:  # pragma: no cover - fallback when mem0 is unavailable
    MemoryClient = None  # type: ignore


@dataclass
class MemoryRecord:
    """Represents a single memory entry."""

    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class MemoryStore:
    """Wraps mem0 if available, otherwise falls back to an in-memory store."""

    def __init__(self, namespace: str = "superagent") -> None:
        self._namespace = namespace
        if MemoryClient is not None:
            self._client = MemoryClient(namespace=namespace)
            self._in_memory: Optional[List[MemoryRecord]] = None
        else:
            self._client = None
            self._in_memory = []

    def write(self, content: str, **metadata: Any) -> None:
        """Persist a memory entry."""

        if not content:
            return
        if self._client is not None:
            self._client.write(content=content, metadata=metadata)
        else:
            assert self._in_memory is not None
            self._in_memory.append(MemoryRecord(content=content, metadata=metadata))

    def search(self, query: str, limit: int = 5) -> Iterable[MemoryRecord]:
        """Retrieve related memories."""

        if self._client is not None:
            results = self._client.search(query=query, limit=limit)
            for item in results:
                yield MemoryRecord(
                    content=item.get("content", ""),
                    metadata=item.get("metadata", {}),
                )
        else:
            assert self._in_memory is not None
            query_lower = query.lower()
            matched = [
                record
                for record in self._in_memory
                if query_lower in record.content.lower()
            ]
            yield from matched[:limit]

    @property
    def namespace(self) -> str:
        return self._namespace
