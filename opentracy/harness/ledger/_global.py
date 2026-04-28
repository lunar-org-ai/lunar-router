"""Process-wide singleton accessor for the ledger store.

Mirrors the `get_memory_store()` pattern in harness/memory_store.py so
schedulers and scanners can reach the ledger without plumbing a store
through every callsite. Tests construct their own `LedgerStore(tmp_path)`
and never touch this global.
"""

from __future__ import annotations

from typing import Optional

from .store import LedgerStore


_instance: Optional[LedgerStore] = None


def get_ledger_store() -> LedgerStore:
    """Return the process-wide LedgerStore, creating it on first access.

    Path resolution lives in `LedgerStore.__init__`:
      1. explicit db_path
      2. OPENTRACY_LEDGER_DB env
      3. ~/.opentracy/harness_ledger.sqlite
    """
    global _instance
    if _instance is None:
        _instance = LedgerStore()
    return _instance


def reset_ledger_store_for_tests() -> None:
    """Clear the singleton so tests can install a tmp-path instance.

    Closes the existing connection if one was opened.
    """
    global _instance
    if _instance is not None:
        try:
            _instance.close()
        except Exception:
            pass
    _instance = None
