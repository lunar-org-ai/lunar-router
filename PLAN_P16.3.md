# PLAN — P16.3 · BYOK + KMS envelope encryption

| Field | Value |
|---|---|
| Phase | P16.3 |
| Parent | P16 (Remote MCP, multi-tenant, BYOK, GCP deploy) |
| Status | Not started |
| Depends on | P16.1 (per-tenant secrets paths) |
| Unblocks | P16.5 (deploy — operator-anthropic-key in Secret Manager + per-tenant DEKs) |
| Reference | standard envelope-encryption pattern (DEK / KEK split) |

## Goal

Encrypt per-tenant BYOK API keys (Anthropic, OpenAI) at rest with **Google Cloud KMS envelope encryption** when deployed on the hosted infra. OSS local users keep the existing plaintext `secrets.env` flow — the new crypto is **opt-in** via an env var (independent of `OPENTRACY_MULTI_TENANT`).

Threat model:
- ✅ At-rest leak: ciphertext on disk / gcsfuse / bucket snapshot is useless without KMS access.
- ❌ In-memory leak: decrypted keys are in process memory by the time the LLM SDK uses them. Out of scope — anyone with process memory access can already MITM the SDK call.
- ❌ In-flight from KMS to runtime: GCP internal network with IAM-bound service account. Documented; not enforced by code.

## Locked decisions

- **Three backends** behind a `Crypto` Protocol:
  - `NoopCrypto` — pass-through. Default. Preserves OSS behavior exactly.
  - `GoogleKmsCrypto` — real envelope encryption via `google-cloud-kms` SDK.
  - `FakeKmsCrypto` — in-memory mock for tests. Lives in `runtime/crypto/fake.py` and is the ONLY one any test ever instantiates directly.
- **Selection**: factory reads env once per-call (so tests can flip per case):
  - `OPENTRACY_KMS_KEY_NAME=projects/.../locations/.../keyRings/opentracy-byok/cryptoKeys/byok-master` → `GoogleKmsCrypto`
  - Otherwise → `NoopCrypto`
- **Storage format change** when KMS is on: the per-agent `secrets.env` plaintext file gets replaced by `secrets.enc.json`:
  ```json
  {
    "v": 1,                       // envelope format version
    "kek": "projects/.../cryptoKeys/byok-master",
    "kek_version": 1,             // mirror of the KMS key version used to wrap DEK
    "encrypted_dek": "<base64>",  // DEK wrapped by KEK
    "nonce": "<base64>",          // AES-GCM nonce (12 bytes)
    "ciphertext": "<base64>"      // AES-GCM(plaintext_env_body, DEK, nonce)
  }
  ```
  The plaintext body inside is the same dotenv-style `KEY=VALUE` we use today — just sealed.
- **Independence from multi-tenant flag**: KMS is *orthogonal* to `OPENTRACY_MULTI_TENANT`. An operator could turn on KMS without going multi-tenant if they want at-rest encryption for a single-tenant deploy. Both flags are independent ON/OFF.
- **Read fallback**: `load_secrets()` prefers `secrets.enc.json` when present. Falls back to `secrets.env` plaintext for OSS / pre-migration installs. Never reads both into the same dict — first-found wins.
- **Write rule**: `save_secrets()` writes whichever format the current `select_crypto()` selects. If migrating from plaintext to KMS, the operator runs `tools/migrate_secrets_to_kms.py` once. The runtime does NOT auto-migrate on the fly (too easy to lose data on a wrong env var).
- **Crypto primitive**: AES-256-GCM via `cryptography` (already in deps via `mcp[security]`? if not, add). 32-byte DEKs, 12-byte nonces, base64-encoded for JSON storage.
- **Key rotation**: NOT in this phase. `kek_version` is stamped in the envelope but the runtime always asks KMS for the *primary* version. Rotation is a separate phase (P16.3.1).
- **Scope cut**: `secrets.env` only. We do NOT encrypt other per-agent files (`onboarding.json`, `integrations/<channel>.json`, etc) — they contain non-secret config. Slack/Twilio signing-secret-in-the-clear is a known follow-up.

## Architecture

```
save_secrets(agent_id, {"ANTHROPIC_API_KEY": "sk-ant-..."})
                                                                   
  ┌─ select_crypto() reads OPENTRACY_KMS_KEY_NAME ─┐                
  │                                                │                
  │  KMS unset → NoopCrypto                        │                
  │    ↓                                           │                
  │    write secrets.env  (KEY=VALUE plaintext)    │                
  │                                                │                
  │  KMS set → GoogleKmsCrypto                     │                
  │    ↓                                           │                
  │    1. generate 32-byte DEK locally             │                
  │    2. encrypt dotenv body with AES-GCM(DEK)    │                
  │    3. call kms.encrypt(KEK, DEK) → wrapped_DEK │                
  │    4. write secrets.enc.json with envelope     │                
  └────────────────────────────────────────────────┘                

load_secrets(agent_id):
  if secrets.enc.json exists:
    1. read envelope
    2. call kms.decrypt(KEK, wrapped_DEK) → DEK
    3. decrypt body with AES-GCM(DEK, nonce)
    4. parse KEY=VALUE
  elif secrets.env exists:
    return parse plaintext  (legacy / OSS path)
  else:
    return {}
```

## File changes

### New

- **`runtime/crypto/__init__.py`** — re-exports.
- **`runtime/crypto/protocol.py`** — `Crypto` Protocol with `encrypt(plaintext: bytes) -> bytes` and `decrypt(blob: bytes) -> bytes`.
- **`runtime/crypto/envelope.py`** — `Envelope` dataclass + JSON ser/de + AES-GCM helpers.
- **`runtime/crypto/noop.py`** — `NoopCrypto`. `encrypt` and `decrypt` return their input.
- **`runtime/crypto/google_kms.py`** — `GoogleKmsCrypto`. Wraps `google-cloud-kms` client; generates DEK locally, wraps via KMS, AES-GCMs the body.
- **`runtime/crypto/fake.py`** — `FakeKmsCrypto`. Uses a static in-memory KEK to wrap a DEK without network. Tests only.
- **`runtime/crypto/factory.py`** — `select_crypto()` reads env, returns the right instance.
- **`tools/migrate_secrets_to_kms.py`** — one-shot script: walks `tenants/<tid>/agents/<aid>/secrets.env`, encrypts, writes `secrets.enc.json`, deletes plaintext.
- **`runtime/tests/test_crypto_envelope.py`** — round-trip + format tests using `FakeKmsCrypto`.
- **`runtime/tests/test_secrets_with_kms.py`** — `save_secrets` / `load_secrets` round-trip with `FakeKmsCrypto` patched in.

### Modified

- **`runtime/agents/secrets.py`**:
  - `load_secrets`: prefer `secrets.enc.json` over `secrets.env` if both exist.
  - `save_secrets`: writes the format selected by `select_crypto()`.
  - Existing `secrets.env` callers stay working (KMS off path returns the legacy behavior byte-for-byte).
- **`pyproject.toml`** — add `cryptography>=42` and `google-cloud-kms>=2.20` to deps. Optional dependency on KMS for OSS: pin `cryptography` only as a base dep; `google-cloud-kms` goes in an `[project.optional-dependencies] kms = […]` extra so the OSS clone doesn't have to pull it.

## Tests

| Layer | What | How |
|---|---|---|
| Unit | Envelope round-trip | encrypt → decrypt yields plaintext; tampered ciphertext raises |
| Unit | NoopCrypto pass-through | `encrypt(x) == x`, `decrypt(x) == x` |
| Unit | FakeKmsCrypto isolates between instances | two instances with different fake KEKs can't decrypt each other |
| Integration | `save_secrets` + `load_secrets` round-trip under FakeKmsCrypto | dotenv → JSON envelope → back to dotenv |
| Integration | Read-side fallback to plaintext when JSON missing | OSS files still load |
| Integration | Write switches format on env-flag change | KMS off → plaintext, KMS on → JSON envelope |
| Integration | Mask + status surface identical under both backends | UI never sees raw keys regardless of storage |
| Tools | Migration script idempotent | running twice doesn't re-encrypt re-encrypted files |

## Risks

| Risk | Mitigation |
|---|---|
| `google-cloud-kms` adds 20+ MB transitive deps; OSS clone explodes. | Move it to `[project.optional-dependencies] kms = […]`. OSS users get `pip install opentracy-new-mode`; infra deploys get `pip install opentracy-new-mode[kms]`. |
| KMS quota / latency: every read of a per-tenant secret = one `kms.decrypt` call. Slow. | Cache decrypted DEKs per (kek_version, encrypted_dek) tuple in-process. P16.3 lands the cache stub; tuning is later. |
| Operator forgets to run migration → multi-tenant deploy has plaintext at-rest until first key edit. | `tools/migrate_secrets_to_kms.py` runs idempotently; document in runbook. |
| Test envs accidentally pick up the operator's real `OPENTRACY_KMS_KEY_NAME` and start calling Google. | The factory checks the value: if it doesn't look like a real KMS resource name (`projects/...`), it raises rather than silently falling back, so a stray env var fails loud in tests. Tests always pass `FakeKmsCrypto` explicitly. |
| AES-GCM nonce reuse if a bug regenerates the same nonce. | `secrets.token_bytes(12)` per encrypt — Python's `secrets` is the right source. Document; don't try to "optimize" by reusing nonces. |
| Crypto code looking shiny but actually broken (custom AES-GCM impl, etc). | We use `cryptography.hazmat.primitives.ciphertext.aead.AESGCM` — the audited standard impl. No hand-rolled crypto. |

## What's NOT in this phase

- **Key rotation** — bumping `kek_version` and re-encrypting old envelopes. Tracked but deferred (own milestone).
- **Cache TTL / eviction** for decrypted DEKs — basic dict cache lands, but no expiry policy.
- **Multi-region failover** for KMS — out of scope.
- **HSM-bound keys** — KMS's default keys are sufficient for P16.3. HSM is a paid tier we don't need yet.
- **Audit logging of every encrypt/decrypt call** — P16.3.1 (also blocked by the audit trail work).
- **Per-tenant KEKs** instead of a single org-wide KEK — single KEK simplifies P16.3; per-tenant requires GCP IAM policy machinery and is a P16.4+ scope.

## Order of work

1. **S1** — `runtime/crypto/` scaffolding (Protocol, NoopCrypto, envelope helpers, factory) + unit tests. NO call sites changed.
2. **S2** — `FakeKmsCrypto` + envelope round-trip tests. Pins the contract.
3. **S3** — `GoogleKmsCrypto` skeleton. No live KMS in tests; we monkeypatch the client. Unit-test the wrapping logic.
4. **S4** — Wire into `runtime/agents/secrets.py`. `load_secrets` prefers JSON envelope; `save_secrets` writes whichever format the factory says.
5. **S5** — `tools/migrate_secrets_to_kms.py` + idempotency tests.
6. **S6** — Docs (`docs/multi-tenant.md` KMS section), `pyproject.toml` optional deps, PR (stacked on P16.2).

Each step is its own commit. S1-S3 are pure-Python additions with no behavior change anywhere; the wiring lands in S4.

## Done when

- With no env config, `secrets.env` is plaintext exactly as today and every existing test still passes.
- With `OPENTRACY_KMS_KEY_NAME=...` set, `save_secrets` writes a `secrets.enc.json` whose contents are AES-GCM ciphertext + a KMS-wrapped DEK; `load_secrets` decrypts on read.
- `tools/migrate_secrets_to_kms.py` converts an existing `tenants/_default/agents/<aid>/secrets.env` to JSON envelope and the runtime keeps working.
- `pyproject.toml` declares `kms` as an optional extra so the OSS install isn't ballooned.
