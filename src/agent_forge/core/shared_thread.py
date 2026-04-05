from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ThreadMessage:
    agent: str
    content: str


class SharedThread:
    """
    A shared message log that agents write to and read from.

    Each agent appends its findings after running. When the next agent runs,
    it receives the full thread as additional context so it can build on
    prior work rather than starting from scratch.

    The orchestrator reads the complete thread at the end to synthesize
    a clean final response for the user.
    """

    def __init__(self) -> None:
        self._messages: list[ThreadMessage] = []

    def add(self, agent: str, content: str) -> None:
        """Append an agent's findings to the thread."""
        self._messages.append(ThreadMessage(agent=agent, content=content))

    def messages(self) -> list[ThreadMessage]:
        return list(self._messages)

    def is_empty(self) -> bool:
        return len(self._messages) == 0

    def to_context(self) -> str:
        """Format the thread as a context block to inject into agent prompts."""
        if self.is_empty():
            return ""
        lines = ["=== Findings from other agents so far ==="]
        for msg in self._messages:
            lines.append(f"[{msg.agent}]:\n{msg.content}")
        lines.append("=== End of shared findings ===")
        return "\n\n".join(lines)

    def __len__(self) -> int:
        return len(self._messages)

    def __repr__(self) -> str:
        return f"<SharedThread messages={len(self._messages)}>"
