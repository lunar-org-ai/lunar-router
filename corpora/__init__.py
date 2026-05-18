"""Corpora — knowledge base for RAG retrieval.

Layout::

    corpora/
      ingested/         # raw uploaded docs (.md, .txt)
      indexed/
        index.faiss     # FAISS IndexFlatIP (cosine via normalized vectors)
        manifest.jsonl  # one row per chunk: {id, text, source, metadata}
      usage_stats/      # per-doc retrieval / citation rates (future)

Module API:
  - ``CorpusStore`` — load / query / save the index.
  - ``ingest(paths)`` — read files → chunk → embed → persist.
"""
