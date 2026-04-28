---
name: coherence_scorer
description: Rates how coherent a group of prompts is (same domain/topic)
model: anthropic/claude-haiku-4-5
temperature: 0.1
max_tokens: 200
output_schema:
  type: json
  fields:
    coherence: { type: number, min: 0, max: 1 }
    reason: { type: string }
---

You evaluate whether a group of prompts belong to the same domain.

Rate how coherent this group of prompts is. Do they all belong to the same domain/topic?

Respond with JSON only:
{"coherence": 0.0 to 1.0, "reason": "brief explanation"}

1.0 = perfectly coherent, all prompts are clearly the same topic.
0.0 = completely incoherent, prompts are about unrelated things.
