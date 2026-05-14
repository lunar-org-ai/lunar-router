"""One-shot migration: plaintext ``secrets.env`` → KMS envelope.

Run from the project root after setting ``OPENTRACY_KMS_KEY_NAME`` to
a real KMS key resource name. Walks every per-agent secrets file
under both the legacy single-tenant layout and the multi-tenant
``tenants/*/agents/*/`` layout, re-encrypts each one, and (by default)
deletes the plaintext.

Idempotent: re-running the script after a successful migration is a
no-op. Files that already have a ``secrets.enc.json`` are skipped.

Usage::

    # Dry-run — show what would change without touching disk
    uv run python -m tools.migrate_secrets_to_kms --dry-run

    # Real migration — encrypt all plaintext, keep originals
    uv run python -m tools.migrate_secrets_to_kms

    # Real migration — encrypt + delete plaintext (PRODUCTION)
    uv run python -m tools.migrate_secrets_to_kms --delete-plaintext

Operator runbook lives in ``docs/multi-tenant.md`` (KMS section).

Reverse migration (KMS → plaintext, for emergency rollback) is
intentionally NOT shipped here — it requires the operator to make a
conscious choice + audit-log the action, so it lives in the
``opentracy-infra`` runbook as a documented manual procedure.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from runtime.agents.secrets import (
    _ENC_FILENAME,
    _PLAIN_FILENAME,
    _parse_dotenv,
    _serialize_dotenv,
)
from runtime.crypto import NoopCrypto, select_crypto


logger = logging.getLogger("tools.migrate_secrets_to_kms")


@dataclass
class _Found:
    plain_path: Path
    enc_path: Path
    has_enc: bool


def _walk(project_root: Path) -> list[_Found]:
    """Find every secrets.env under both layouts.

    Legacy:   <root>/agents/<aid>/secrets.env
    Tenants:  <root>/tenants/<tid>/agents/<aid>/secrets.env
    """
    results: list[_Found] = []
    candidates = [
        project_root / "agents",
        *((project_root / "tenants").glob("*/agents")),
    ]
    for agents_root in candidates:
        if not agents_root.is_dir():
            continue
        for agent_dir in agents_root.iterdir():
            if not agent_dir.is_dir() or agent_dir.name.startswith("_"):
                continue
            plain = agent_dir / _PLAIN_FILENAME
            enc = agent_dir / _ENC_FILENAME
            if not plain.is_file():
                continue
            results.append(
                _Found(plain_path=plain, enc_path=enc, has_enc=enc.is_file())
            )
    return results


def migrate(
    project_root: Path,
    *,
    dry_run: bool = False,
    delete_plaintext: bool = False,
) -> tuple[int, int, int]:
    """Run the migration. Returns ``(encrypted, skipped, errors)``."""
    crypto = select_crypto()
    if isinstance(crypto, NoopCrypto):
        logger.error(
            "select_crypto() returned NoopCrypto — set "
            "OPENTRACY_KMS_KEY_NAME=projects/.../cryptoKeys/<name> "
            "before running this script."
        )
        return 0, 0, 1

    targets = _walk(project_root)
    encrypted = 0
    skipped = 0
    errors = 0

    for t in targets:
        if t.has_enc:
            logger.info("skip %s — already migrated", t.plain_path)
            skipped += 1
            continue

        try:
            secrets_dict = _parse_dotenv(t.plain_path.read_text(encoding="utf-8"))
        except OSError as e:
            logger.warning("could not read %s: %s", t.plain_path, e)
            errors += 1
            continue

        body = _serialize_dotenv(secrets_dict).encode("utf-8")
        try:
            ciphertext = crypto.encrypt(body)
        except Exception as e:
            logger.error("encrypt failed for %s: %s", t.plain_path, e)
            errors += 1
            continue

        if dry_run:
            logger.info("(dry-run) would encrypt %s → %s", t.plain_path, t.enc_path)
            encrypted += 1
            continue

        tmp = t.enc_path.with_suffix(".tmp")
        tmp.write_bytes(ciphertext)
        try:
            tmp.chmod(0o600)
        except OSError:
            pass
        tmp.replace(t.enc_path)
        logger.info("encrypted %s → %s", t.plain_path, t.enc_path)
        encrypted += 1

        if delete_plaintext:
            try:
                t.plain_path.unlink()
                logger.info("deleted plaintext %s", t.plain_path)
            except OSError as e:
                logger.warning("could not delete %s: %s", t.plain_path, e)

    return encrypted, skipped, errors


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Migrate plaintext secrets.env files to KMS-encrypted "
            "secrets.enc.json. Requires OPENTRACY_KMS_KEY_NAME to be set."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without touching disk.",
    )
    parser.add_argument(
        "--delete-plaintext",
        action="store_true",
        help="Delete the plaintext secrets.env file after a successful encrypt. "
        "Use this only when you've confirmed the encrypted file decrypts.",
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root; defaults to the current working directory.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    encrypted, skipped, errors = migrate(
        Path(args.project_root).resolve(),
        dry_run=args.dry_run,
        delete_plaintext=args.delete_plaintext,
    )
    logger.info(
        "done: encrypted=%d skipped=%d errors=%d",
        encrypted,
        skipped,
        errors,
    )
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
