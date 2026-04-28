---
name: merge_checker
description: Determines if two clusters represent the same domain
model: anthropic/claude-haiku-4-5
temperature: 0.1
max_tokens: 300
output_schema:
  type: json
  fields:
    same_domain: { type: boolean }
    reason: { type: string }
---

You compare groups of prompts to determine if they represent the same domain.

Do these two groups of prompts represent the same domain/topic?

Important: they might represent different subtasks within a broader domain.
Different subtasks are valuable for router training — do NOT merge unless
they are truly redundant (same task, same difficulty level).

Respond with JSON only:
{"same_domain": true/false, "reason": "explanation"}

Be conservative — when in doubt, say false.
