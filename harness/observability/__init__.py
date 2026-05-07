"""harness.observability — distillation of trajectories into structured corpus.

Following AHE (Lin et al., 2026): raw traces are not consumed directly. They
get reduced into Sessions (one experiment trajectory) and Epochs (time-
bounded summaries), which the introspection LLM and the UI consume cheaply.
"""
