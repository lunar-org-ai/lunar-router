"""OpenTracy router (UniRoute, autonomous training).

Inspired by UniRoute (arxiv 2502.08773) for the decision rule and
AutoHarness (arxiv 2603.03329) for the self-improvement loop.

This package is structured around four sub-trees:
- core/: math primitives — embeddings, clustering, metrics.
- models/: LLMProfile (Psi vector), LLMRegistry, LLMClient surfaces.
- training/: KMeans trainer + first-fit gate (P15.3.3).
- feedback/, augmentation/, evaluation/: trace -> training data -> Psi update -> eval.

P15.3.1 ships only the core/ + models/ scaffolding. Later phases fill the rest.
"""
