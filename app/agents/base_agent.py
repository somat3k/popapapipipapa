"""Base agent framework: BaseAgent, AgentRegistry, MessageBus, ToolRegistry."""

from __future__ import annotations

import abc
import enum
import logging
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class AgentState(enum.Enum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    ERROR = "ERROR"
    STOPPED = "STOPPED"


class AgentContext:
    """Shared mutable context passed between agents in a pipeline."""

    def __init__(self) -> None:
        self._data: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(key, default)

    def all(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._data)


class ToolRegistry:
    """Registry for agent tools with dynamic registration."""

    def __init__(self) -> None:
        self._tools: Dict[str, Callable[..., Any]] = {}

    def register(self, name: str, fn: Callable[..., Any]) -> None:
        if name in self._tools:
            raise ValueError(f"Tool '{name}' is already registered.")
        self._tools[name] = fn
        logger.debug("Tool registered: %s", name)

    def call(self, name: str, **kwargs: Any) -> Any:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: '{name}'")
        return self._tools[name](**kwargs)

    def list_tools(self) -> List[str]:
        return list(self._tools.keys())


# Module-level shared tool registry
TOOL_REGISTRY = ToolRegistry()


class MessageBus:
    """Simple publish/subscribe message bus for inter-agent communication."""

    def __init__(self) -> None:
        self._subscribers: Dict[str, List[Callable[[Any], None]]] = {}
        self._lock = threading.Lock()

    def subscribe(self, topic: str, callback: Callable[[Any], None]) -> None:
        with self._lock:
            self._subscribers.setdefault(topic, []).append(callback)

    def publish(self, topic: str, message: Any) -> None:
        with self._lock:
            callbacks = list(self._subscribers.get(topic, []))
        for cb in callbacks:
            try:
                cb(message)
            except Exception:
                logger.exception("Error in subscriber for topic '%s'", topic)

    def unsubscribe(self, topic: str, callback: Callable[[Any], None]) -> None:
        with self._lock:
            subs = self._subscribers.get(topic, [])
            if callback in subs:
                subs.remove(callback)


# Module-level shared message bus
MESSAGE_BUS = MessageBus()


class BaseAgent(abc.ABC):
    """Abstract base class for all Multiplex Financials agents.

    Subclasses must implement the ``_execute`` method which contains the
    agent's core logic. The lifecycle is managed by ``run()`` / ``stop()`` /
    ``reset()``.
    """

    def __init__(
        self,
        agent_id: Optional[str] = None,
        name: str = "BaseAgent",
        context: Optional[AgentContext] = None,
        tool_registry: Optional[ToolRegistry] = None,
        message_bus: Optional[MessageBus] = None,
    ) -> None:
        self.agent_id: str = agent_id or str(uuid.uuid4())
        self.name = name
        self.context = context or AgentContext()
        self.tools = tool_registry or TOOL_REGISTRY
        self.bus = message_bus or MESSAGE_BUS
        self._state = AgentState.IDLE
        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_time: Optional[float] = None
        self._error: Optional[Exception] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> AgentState:
        with self._state_lock:
            return self._state

    @state.setter
    def state(self, new_state: AgentState) -> None:
        with self._state_lock:
            old = self._state
            self._state = new_state
        logger.info("[%s] state: %s → %s", self.name, old.value, new_state.value)
        self.bus.publish(f"agent.{self.agent_id}.state", new_state)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run(self, background: bool = True) -> None:
        """Start the agent. If *background* is True, run in a daemon thread."""
        if self.state not in (AgentState.IDLE, AgentState.STOPPED, AgentState.ERROR):
            logger.warning("[%s] Cannot start from state %s", self.name, self.state)
            return
        self._stop_event.clear()
        self._start_time = time.perf_counter()
        self.state = AgentState.RUNNING
        if background:
            self._thread = threading.Thread(
                target=self._safe_execute, daemon=True, name=self.name
            )
            self._thread.start()
        else:
            self._safe_execute()

    def stop(self) -> None:
        """Signal the agent to stop gracefully."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        self.state = AgentState.STOPPED

    def pause(self) -> None:
        """Pause the agent (custom logic may check _stop_event)."""
        self.state = AgentState.PAUSED

    def reset(self) -> None:
        """Reset agent to IDLE after ERROR or STOPPED."""
        self._error = None
        self._stop_event.clear()
        self.state = AgentState.IDLE

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _safe_execute(self) -> None:
        try:
            self._execute()
        except Exception as exc:
            self._error = exc
            logger.exception("[%s] Unhandled error", self.name)
            self.state = AgentState.ERROR
        else:
            if self.state == AgentState.RUNNING:
                self.state = AgentState.IDLE

    @abc.abstractmethod
    def _execute(self) -> None:
        """Core agent logic. Subclasses must implement this."""

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def should_stop(self) -> bool:
        return self._stop_event.is_set()

    def elapsed(self) -> float:
        if self._start_time is None:
            return 0.0
        return time.perf_counter() - self._start_time

    def use_tool(self, tool_name: str, **kwargs: Any) -> Any:
        logger.debug("[%s] Using tool '%s' with %s", self.name, tool_name, kwargs)
        return self.tools.call(tool_name, **kwargs)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} id={self.agent_id[:8]} state={self.state.value}>"

    def __str__(self) -> str:
        return f"{self.name}[{self.state.value}]"


class AgentRegistry:
    """Singleton registry for all running agents."""

    _instance: Optional["AgentRegistry"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "AgentRegistry":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._agents: Dict[str, BaseAgent] = {}
        return cls._instance

    def register(self, agent: BaseAgent) -> None:
        self._agents[agent.agent_id] = agent
        logger.info("Registered agent: %s (%s)", agent.name, agent.agent_id[:8])

    def unregister(self, agent_id: str) -> None:
        self._agents.pop(agent_id, None)

    def get(self, agent_id: str) -> Optional[BaseAgent]:
        return self._agents.get(agent_id)

    def list_agents(self) -> List[BaseAgent]:
        return list(self._agents.values())

    def stop_all(self) -> None:
        for agent in self.list_agents():
            if agent.state == AgentState.RUNNING:
                agent.stop()
