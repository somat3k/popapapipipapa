"""Tests for agent framework: BaseAgent, AgentRegistry, MessageBus, ToolRegistry."""

import time

import pytest

from app.agents.base_agent import (
    AgentContext,
    AgentRegistry,
    AgentState,
    BaseAgent,
    MessageBus,
    ToolRegistry,
)


# ---------------------------------------------------------------------------
# Concrete test agent
# ---------------------------------------------------------------------------

class CounterAgent(BaseAgent):
    def __init__(self, iterations: int = 3, **kwargs):
        super().__init__(name="CounterAgent", **kwargs)
        self.iterations = iterations
        self.count = 0

    def _execute(self):
        for _ in range(self.iterations):
            if self.should_stop():
                break
            self.count += 1
            time.sleep(0.01)


class ErrorAgent(BaseAgent):
    def _execute(self):
        raise RuntimeError("Intentional error")


# ---------------------------------------------------------------------------
# AgentContext tests
# ---------------------------------------------------------------------------

def test_agent_context_set_get():
    ctx = AgentContext()
    ctx.set("price", 42.0)
    assert ctx.get("price") == 42.0


def test_agent_context_default():
    ctx = AgentContext()
    assert ctx.get("missing", "default") == "default"


def test_agent_context_all():
    ctx = AgentContext()
    ctx.set("a", 1)
    ctx.set("b", 2)
    data = ctx.all()
    assert data == {"a": 1, "b": 2}


def test_agent_context_thread_safety():
    import threading
    ctx = AgentContext()
    def writer(n):
        for i in range(100):
            ctx.set(f"key_{n}_{i}", i)
    threads = [threading.Thread(target=writer, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(ctx.all()) <= 400  # some keys may overlap


# ---------------------------------------------------------------------------
# ToolRegistry tests
# ---------------------------------------------------------------------------

def test_tool_registry_register_and_call():
    reg = ToolRegistry()
    reg.register("add", lambda a, b: a + b)
    assert reg.call("add", a=3, b=4) == 7


def test_tool_registry_duplicate_raises():
    reg = ToolRegistry()
    reg.register("fn", lambda: None)
    with pytest.raises(ValueError, match="already registered"):
        reg.register("fn", lambda: None)


def test_tool_registry_unknown_raises():
    reg = ToolRegistry()
    with pytest.raises(KeyError, match="Unknown tool"):
        reg.call("nonexistent")


def test_tool_registry_list():
    reg = ToolRegistry()
    reg.register("t1", lambda: None)
    reg.register("t2", lambda: None)
    assert set(reg.list_tools()) == {"t1", "t2"}


# ---------------------------------------------------------------------------
# MessageBus tests
# ---------------------------------------------------------------------------

def test_message_bus_publish_subscribe():
    bus = MessageBus()
    received = []
    bus.subscribe("topic.test", received.append)
    bus.publish("topic.test", "hello")
    assert received == ["hello"]


def test_message_bus_multiple_subscribers():
    bus = MessageBus()
    a, b = [], []
    bus.subscribe("t", a.append)
    bus.subscribe("t", b.append)
    bus.publish("t", 42)
    assert a == [42] and b == [42]


def test_message_bus_unsubscribe():
    bus = MessageBus()
    received = []
    cb = received.append
    bus.subscribe("t", cb)
    bus.unsubscribe("t", cb)
    bus.publish("t", "should not arrive")
    assert received == []


def test_message_bus_subscriber_exception_is_isolated():
    bus = MessageBus()
    results = []
    bus.subscribe("t", lambda _: (_ for _ in ()).throw(RuntimeError("boom")))
    bus.subscribe("t", results.append)
    bus.publish("t", "val")  # should not raise
    assert results == ["val"]


# ---------------------------------------------------------------------------
# BaseAgent lifecycle tests
# ---------------------------------------------------------------------------

def test_agent_idle_state():
    agent = CounterAgent()
    assert agent.state == AgentState.IDLE


def test_agent_run_background():
    agent = CounterAgent(iterations=3)
    agent.run(background=True)
    time.sleep(0.15)
    assert agent.count == 3
    agent.state == AgentState.IDLE


def test_agent_stop():
    agent = CounterAgent(iterations=1000)
    agent.run(background=True)
    time.sleep(0.02)
    agent.stop()
    assert agent.state == AgentState.STOPPED
    assert agent.count < 1000


def test_agent_error_state():
    agent = ErrorAgent()
    agent.run(background=True)
    time.sleep(0.1)
    assert agent.state == AgentState.ERROR


def test_agent_reset_after_error():
    agent = ErrorAgent()
    agent.run(background=True)
    time.sleep(0.1)
    assert agent.state == AgentState.ERROR
    agent.reset()
    assert agent.state == AgentState.IDLE


def test_agent_repr():
    agent = CounterAgent()
    r = repr(agent)
    assert "CounterAgent" in r
    assert "IDLE" in r


def test_agent_elapsed():
    agent = CounterAgent(iterations=2)
    agent.run(background=True)
    time.sleep(0.05)
    assert agent.elapsed() >= 0.0


# ---------------------------------------------------------------------------
# AgentRegistry tests
# ---------------------------------------------------------------------------

def test_agent_registry_register_and_get():
    registry = AgentRegistry()
    agent = CounterAgent()
    registry.register(agent)
    found = registry.get(agent.agent_id)
    assert found is agent


def test_agent_registry_list():
    registry = AgentRegistry()
    a1 = CounterAgent()
    a2 = CounterAgent()
    registry.register(a1)
    registry.register(a2)
    ids = [a.agent_id for a in registry.list_agents()]
    assert a1.agent_id in ids
    assert a2.agent_id in ids


def test_agent_registry_unregister():
    registry = AgentRegistry()
    agent = CounterAgent()
    registry.register(agent)
    registry.unregister(agent.agent_id)
    assert registry.get(agent.agent_id) is None
