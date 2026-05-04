#!/usr/bin/env python3
"""
Seed 300 realistic fake LLM traces into ClickHouse for frontend/harness testing.

Usage:
    python scripts/seed_traces.py [--count 300]

Environment (defaults to local dev stack):
    OPENTRACY_CH_ENABLED=true
    OPENTRACY_CH_HOST=localhost
    OPENTRACY_CH_HTTP_PORT=8123
    OPENTRACY_CH_DATABASE=opentracy
    OPENTRACY_CH_USERNAME=default
    OPENTRACY_CH_PASSWORD=opentracy
"""

from __future__ import annotations

import argparse
import os
import random
import uuid
from datetime import datetime, timedelta, timezone

import clickhouse_connect

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CH_HOST     = os.getenv("OPENTRACY_CH_HOST",      "localhost")
CH_PORT     = int(os.getenv("OPENTRACY_CH_HTTP_PORT", "8123"))
CH_DB       = os.getenv("OPENTRACY_CH_DATABASE",  "opentracy")
CH_USER     = os.getenv("OPENTRACY_CH_USERNAME",  "default")
CH_PASSWORD = os.getenv("OPENTRACY_CH_PASSWORD",  "opentracy")

# ---------------------------------------------------------------------------
# Models with real prices (subset from model_prices.py, varied providers)
# (input_cost_per_token, output_cost_per_token, provider)
# ---------------------------------------------------------------------------
MODELS: list[tuple[str, float, float, str]] = [
    # OpenAI
    ("gpt-4o",              2.50e-6, 10.00e-6, "openai"),
    ("gpt-4o-mini",         0.15e-6,  0.60e-6, "openai"),
    ("gpt-4-turbo",        10.00e-6, 30.00e-6, "openai"),
    ("gpt-3.5-turbo",       0.50e-6,  1.50e-6, "openai"),
    ("o1-mini",             3.00e-6, 12.00e-6, "openai"),
    ("o3-mini",             1.10e-6,  4.40e-6, "openai"),
    # Anthropic
    ("claude-sonnet-4-20250514",     3.00e-6,  15.00e-6, "anthropic"),
    ("claude-3-haiku-20240307",      0.25e-6,   1.25e-6, "anthropic"),
    ("claude-opus-4-20250514",      15.00e-6,  75.00e-6, "anthropic"),
    # Gemini
    ("gemini-2.0-flash",    0.10e-6,  0.40e-6, "google"),
    ("gemini-1.5-pro",      1.25e-6,  5.00e-6, "google"),
    # Mistral
    ("mistral-large-latest",  2.00e-6, 6.00e-6, "mistral"),
    ("mistral-small-latest",  0.20e-6, 0.60e-6, "mistral"),
    # DeepSeek
    ("deepseek-chat",       0.14e-6,  0.28e-6, "deepseek"),
    ("deepseek-reasoner",   0.55e-6,  2.19e-6, "deepseek"),
    # Groq
    ("llama-3.3-70b-versatile", 0.59e-6, 0.79e-6, "groq"),
    ("llama-3.1-8b-instant",    0.05e-6, 0.08e-6, "groq"),
]

# Realistic prompt / response pairs for diverse trace content
SAMPLE_EXCHANGES: list[tuple[str, str]] = [
    ("Explain quantum entanglement in simple terms.",
     "Quantum entanglement is a phenomenon where two particles become linked so that measuring one instantly affects the other, regardless of distance."),
    ("Write a Python function to reverse a linked list.",
     "def reverse(head):\n    prev = None\n    curr = head\n    while curr:\n        nxt = curr.next\n        curr.next = prev\n        prev = curr\n        curr = nxt\n    return prev"),
    ("Summarize the key ideas from 'The Lean Startup'.",
     "The Lean Startup advocates building minimum viable products, measuring outcomes, and iterating quickly based on validated learning to reduce waste and risk."),
    ("What are the OWASP Top 10 vulnerabilities?",
     "The OWASP Top 10 includes: Broken Access Control, Cryptographic Failures, Injection, Insecure Design, Security Misconfiguration, Vulnerable Components, Auth Failures, Data Integrity Failures, Security Logging Failures, SSRF."),
    ("Translate 'The weather is beautiful today' to French.",
     "Le temps est magnifique aujourd'hui."),
    ("Generate a regex to match email addresses.",
     r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"),
    ("What is the difference between SQL and NoSQL databases?",
     "SQL databases use structured schemas and support ACID transactions; NoSQL databases offer flexible schemas and horizontal scalability, trading some consistency guarantees for performance."),
    ("Write unit tests for a calculator class in Python.",
     "import unittest\nclass TestCalculator(unittest.TestCase):\n    def test_add(self): self.assertEqual(Calculator().add(2, 3), 5)\n    def test_div_by_zero(self): self.assertRaises(ZeroDivisionError, Calculator().divide, 1, 0)"),
    ("How does the transformer architecture work?",
     "Transformers use self-attention to relate tokens in a sequence to each other in parallel, enabling efficient long-range dependency modeling without recurrence."),
    ("Generate a marketing email for a new SaaS product launch.",
     "Subject: Introducing OpenTracy — LLM observability made simple.\n\nHi [Name],\n\nWe're excited to launch OpenTracy, your open-source gateway for routing, tracing, and optimizing LLM calls. Sign up free today."),
    ("What is a vector database and when should I use one?",
     "A vector database stores high-dimensional embeddings and enables fast similarity search. Use it when building semantic search, RAG pipelines, or recommendation systems."),
    ("Explain the CAP theorem.",
     "CAP states that a distributed system can only guarantee two of three: Consistency, Availability, Partition Tolerance. Most systems choose CP or AP."),
    ("Write a bash script to monitor disk usage and send an alert.",
     "#!/bin/bash\nTHRESHOLD=90\nUSAGE=$(df / | awk 'NR==2{print $5}' | tr -d '%')\n[ $USAGE -gt $THRESHOLD ] && echo 'Disk alert: '$USAGE'%' | mail -s 'Disk Warning' admin@example.com"),
    ("How do I implement JWT authentication in FastAPI?",
     "Use python-jose for token creation and python-multipart for form parsing. Create /token endpoint returning JWT, then use OAuth2PasswordBearer as a dependency in protected routes."),
    ("What are the best practices for prompt engineering?",
     "Be specific and clear, provide examples (few-shot), use role prompting, break complex tasks into steps, and iterate based on outputs."),
    ("Describe the microservices architecture pattern.",
     "Microservices decompose an application into small, independently deployable services communicating via APIs, enabling independent scaling and technology diversity at the cost of distributed systems complexity."),
    ("How does retrieval-augmented generation (RAG) work?",
     "RAG retrieves relevant documents from a knowledge base using vector similarity, then injects them into the LLM prompt as context, improving accuracy and reducing hallucinations."),
    ("Write a SQL query to find the top 5 customers by revenue.",
     "SELECT customer_id, SUM(amount) AS revenue\nFROM orders\nGROUP BY customer_id\nORDER BY revenue DESC\nLIMIT 5;"),
    ("What is the difference between TCP and UDP?",
     "TCP is connection-oriented with guaranteed delivery and ordering; UDP is connectionless, faster, but without delivery guarantees — suited for streaming and gaming."),
    ("Explain gradient descent and its variants.",
     "Gradient descent minimizes a loss function by iteratively stepping in the direction of steepest descent. Variants include SGD (single sample), mini-batch GD, Adam (adaptive moments), and RMSProp."),
]

ERROR_MESSAGES = [
    ("rate_limit_exceeded", "upstream", "Rate limit reached for model. Please retry after 60s."),
    ("context_length_exceeded", "client",  "This model's maximum context length is 128000 tokens."),
    ("model_not_available", "unavailable", "The model is temporarily unavailable. Please try again."),
    ("invalid_api_key", "client",  "Incorrect API key provided."),
    ("", "", ""),  # success (weight towards success)
    ("", "", ""),
    ("", "", ""),
    ("", "", ""),
    ("", "", ""),
]


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _random_timestamp(days_back: int = 14) -> datetime:
    """Random timestamp within the last `days_back` days."""
    delta = timedelta(
        seconds=random.randint(0, days_back * 86400)
    )
    return _now_utc() - delta


def _build_row(rng: random.Random) -> list:
    model, in_price, out_price, provider = rng.choice(MODELS)

    tokens_in  = rng.randint(50, 2000)
    tokens_out = rng.randint(20, 800)
    total_tokens = tokens_in + tokens_out

    latency_ms = rng.uniform(200, 8000)
    ttft_ms    = rng.uniform(80, min(latency_ms, 2000))

    input_cost  = tokens_in  * in_price
    output_cost = tokens_out * out_price
    total_cost  = input_cost + output_cost

    err_msg, err_cat, err_text = rng.choice(ERROR_MESSAGES)
    is_error = 1 if err_msg else 0

    exchange = rng.choice(SAMPLE_EXCHANGES)
    input_text  = exchange[0]
    output_text = exchange[1] if not is_error else ""

    input_messages  = f'[{{"role":"user","content":"{input_text[:200]}"}}]'
    output_message  = f'{{"role":"assistant","content":"{output_text[:200]}"}}'

    request_types  = ["chat", "chat_stream", "route"]
    request_type   = rng.choices(request_types, weights=[6, 3, 1])[0]
    is_stream      = 1 if request_type == "chat_stream" else 0
    finish_reason  = "stop" if not is_error else "error"
    tokens_per_s   = tokens_out / (latency_ms / 1000) if latency_ms > 0 else 0

    cluster_id     = rng.randint(0, 8)
    fallback_count = rng.choices([0, 1, 2], weights=[85, 12, 3])[0]

    timestamp = _random_timestamp(days_back=14)

    return [
        str(uuid.uuid4()),          # request_id
        timestamp,                   # timestamp
        model,                       # selected_model
        provider,                    # provider
        cluster_id,                  # cluster_id
        rng.uniform(0, 0.3),        # expected_error
        rng.uniform(-1, 1),         # cost_adjusted_score
        latency_ms,                  # latency_ms
        ttft_ms,                     # ttft_ms
        rng.uniform(0, 5),          # routing_ms
        rng.uniform(0, 20),         # embedding_ms
        tokens_in,                   # tokens_in
        tokens_out,                  # tokens_out
        total_tokens,                # total_tokens
        input_cost,                  # input_cost_usd
        output_cost,                 # output_cost_usd
        0.0,                         # cache_input_cost_usd
        total_cost,                  # total_cost_usd
        is_error,                    # is_error
        err_cat,                     # error_category
        err_text,                    # error_message
        request_type,                # request_type
        is_stream,                   # is_stream
        0,                           # cache_hit
        fallback_count,              # fallback_count
        "[]",                        # provider_attempts
        "{}",                        # all_scores
        "[]",                        # cluster_probabilities
        input_text[:4000],           # input_text
        output_text[:4000],          # output_text
        input_messages[:8000],       # input_messages
        output_message[:4000],       # output_message
        finish_reason,               # finish_reason
        tokens_per_s,               # tokens_per_s
    ]


COLUMN_NAMES = [
    "request_id", "timestamp", "selected_model", "provider",
    "cluster_id", "expected_error", "cost_adjusted_score",
    "latency_ms", "ttft_ms", "routing_ms", "embedding_ms",
    "tokens_in", "tokens_out", "total_tokens",
    "input_cost_usd", "output_cost_usd", "cache_input_cost_usd", "total_cost_usd",
    "is_error", "error_category", "error_message",
    "request_type", "is_stream", "cache_hit",
    "fallback_count", "provider_attempts", "all_scores", "cluster_probabilities",
    "input_text", "output_text", "input_messages", "output_message",
    "finish_reason", "tokens_per_s",
]


def main(count: int = 300, clear: bool = True) -> None:
    print(f"Connecting to ClickHouse at {CH_HOST}:{CH_PORT} …")
    client = clickhouse_connect.get_client(
        host=CH_HOST,
        port=CH_PORT,
        database=CH_DB,
        username=CH_USER,
        password=CH_PASSWORD,
    )

    if clear:
        print("Truncating llm_traces …")
        client.command("TRUNCATE TABLE llm_traces")
        print("  done.")

    rng = random.Random(42)
    rows = [_build_row(rng) for _ in range(count)]

    print(f"Inserting {count} fake traces …")
    client.insert("llm_traces", rows, column_names=COLUMN_NAMES)
    print(f"  {count} rows inserted.")

    # Summary
    total = client.command("SELECT count() FROM llm_traces")
    print(f"\nllm_traces now has {total} rows total.")

    model_counts = client.query(
        "SELECT selected_model, count() AS n FROM llm_traces "
        "GROUP BY selected_model ORDER BY n DESC"
    )
    print("\nDistribution by model:")
    for row in model_counts.result_rows:
        print(f"  {row[0]:<45} {row[1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed fake LLM traces for testing.")
    parser.add_argument("--count", type=int, default=300, help="Number of traces to insert (default: 300)")
    parser.add_argument("--no-clear", dest="clear", action="store_false", help="Skip truncating existing data")
    args = parser.parse_args()
    main(count=args.count, clear=args.clear)
