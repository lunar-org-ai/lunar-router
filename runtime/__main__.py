"""CLI entry: `uv run python -m runtime "your question"`.

Loads agent/agent.yaml, compiles the pipeline, runs the request, prints the
response on stdout and a per-stage summary + trace_id on stderr, and persists
the full trace to traces/raw/.
"""

from __future__ import annotations

import argparse
import sys

from runtime.compiler.builder import compile_agent
from runtime.compiler.loader import load_agent
from runtime.executor.pipeline import PipelineExecutor
from runtime.executor.tracing import write_trace


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="runtime", description="Run the agent on a single request"
    )
    parser.add_argument("request", help="The user request")
    parser.add_argument(
        "--agent",
        default="agent/agent.yaml",
        help="Path to agent.yaml (default: agent/agent.yaml)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print only the response (no per-stage summary)",
    )
    args = parser.parse_args(argv)

    cfg = load_agent(args.agent)
    pipe = compile_agent(cfg)
    exe = PipelineExecutor(pipe)

    _, rec = exe.run(args.request)
    trace_id = write_trace(rec)

    if not args.quiet:
        print(f"--- agent {cfg.version} ---", file=sys.stderr)
        for s in rec.stages:
            err = f" ⚠ {s.error}" if s.error else ""
            print(
                f"  {s.stage:>9}  {s.duration_ms:>7.2f}ms  "
                f"{s.technique}/{s.variant}  docs {s.docs_in}→{s.docs_out}{err}",
                file=sys.stderr,
            )
        print(f"  total:    {rec.duration_ms:>7.2f}ms", file=sys.stderr)
        print(f"  trace:    {trace_id}", file=sys.stderr)
        print(file=sys.stderr)

    if not rec.success:
        print(f"ERROR: {rec.error}", file=sys.stderr)
        return 1

    print(rec.response or "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
