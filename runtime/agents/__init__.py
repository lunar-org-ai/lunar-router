"""Multi-agent foundation (P2.0).

Layout::

    agents/
      registry.json         # index + active pointer
      _default/             # the agent that pre-existed multi-agent (migrated)
        agent.yaml
        prompts/system.md
        pipeline/*.yaml
        versions/v<n>/      # snapshots taken by the harness
        onboarding.json     # the day-0 config that birthed it (if any)
      <new_agent_id>/       # each onboarding creates one
        ...

    agent/                  # LIVE running agent — equals the active one
      ...                   # contents copied from agents/<active>/

When the operator switches active agents, ``agents/<id>/`` is copied
into ``agent/`` and the runtime pipeline is recompiled. Traces and
lessons accumulated under the old agent are tagged with their
``agent_id`` (record-level partitioning); directory-level partitioning
is a follow-up.

Public API:
  - ``runtime.agents.registry`` — CRUD on the catalog.
  - ``runtime.agents.types``    — AgentMetadata dataclass.
"""
