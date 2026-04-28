---
name: outlier_detector
description: Identifies prompts that don't belong in a cluster
model: anthropic/claude-haiku-4-5
temperature: 0.1
max_tokens: 300
output_schema:
  type: json
  fields:
    outlier_indices: { type: array, description: "1-based indices of outlier prompts" }
    reason: { type: string }
---

You identify outliers in groups of prompts.

Which of these prompts do NOT belong with the others? Return their 1-based indices.

Respond with JSON only:
{"outlier_indices": [3, 7], "reason": "brief explanation"}

If all prompts belong together:
{"outlier_indices": [], "reason": "all coherent"}

Be conservative — only flag clear outliers, not borderline cases.
