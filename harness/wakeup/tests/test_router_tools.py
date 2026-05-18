"""Tests for harness.introspection.lib.{router_health_check, propose_router_retrain}."""

from __future__ import annotations

import pytest

from harness.introspection.lib import (
    propose_router_retrain,
    router_health_check,
)


def test_router_health_check_cold_start():
    h = router_health_check()
    assert isinstance(h, dict)
    assert h["cold_start"] is True
    assert h["version"] is None
    assert h["k"] is None


def test_propose_blocked_by_policy_off(monkeypatch):
    """Policy mode 'off' for router_config → blocked without invoking the proposer."""

    class FakePolicy:
        @staticmethod
        def from_yaml():
            class P:
                @staticmethod
                def mode_for(kind):
                    return "off"
            return P()

    monkeypatch.setattr("harness.approver.policy.Policy", FakePolicy)
    out = propose_router_retrain(rationale="testing block")
    assert out["action"] == "blocked"
    assert "policy" in out["reason"].lower()


def test_propose_blocked_by_policy_load_error(monkeypatch):
    def boom():
        raise OSError("policy.yaml unreadable")

    class FakePolicy:
        @staticmethod
        def from_yaml():
            boom()

    monkeypatch.setattr("harness.approver.policy.Policy", FakePolicy)
    out = propose_router_retrain(rationale="x")
    assert out["action"] == "blocked"
    assert "policy_load_error" in out["reason"]
