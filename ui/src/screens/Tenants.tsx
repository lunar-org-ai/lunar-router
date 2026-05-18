/**
 * Tenants — operator admin screen (P16.4).
 *
 * Lists every tenant under ``tenants/``, lets the operator create
 * + delete tenants, and drill into one to mint / revoke per-tenant
 * Bearer tokens. Only visible when the runtime is in multi-tenant
 * mode (``OPENTRACY_MULTI_TENANT=1``); the sidebar nav hides
 * otherwise, gated by ``getFeatures()``.
 *
 * Token mint shows the plaintext ONCE on a banner with a copy-to-
 * clipboard button. After dismiss, the value is gone.
 */

import { useCallback, useEffect, useState } from 'react';
import { Icon } from '../components/Icon';
import { Button } from '../components/ui/button';
import { Card } from '../components/ui/card';
import { Input } from '../components/ui/input';
import {
  ApiError,
  createTenant,
  deleteTenant,
  listTenants,
  listTokens,
  mintToken,
  revokeToken,
  type TenantSummary,
  type TokenMintResponse,
  type TokenSummary,
} from '../api';


type Stage = 'list' | 'creating';


export const Tenants = () => {
  const [tenants, setTenants] = useState<TenantSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [stage, setStage] = useState<Stage>('list');
  const [selected, setSelected] = useState<string | null>(null);

  const load = useCallback(async () => {
    setError(null);
    try {
      const res = await listTenants();
      setTenants(res.tenants);
    } catch (e) {
      setError(
        e instanceof ApiError
          ? `Backend ${e.status}: ${e.message}`
          : `Network error: ${e instanceof Error ? e.message : String(e)}`,
      );
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  return (
    <div className="content">
      <h1 className="page-title">Tenants</h1>
      <p className="page-sub">
        Each tenant is an isolated org with its own agents, ledger, traces, and Bearer tokens.
        OSS local installs run without tenants — this screen is only reachable when{' '}
        <code>OPENTRACY_MULTI_TENANT=1</code> is set on the runtime.
      </p>

      {error && (
        <Card className="mb-4 border-destructive p-4">
          <p className="dim m-0 text-destructive">{error}</p>
        </Card>
      )}

      <Card className="mb-6 gap-0 py-0">
        <div className="flex items-center justify-between border-b border-border px-4 py-3.5">
          <div className="text-[13px] font-semibold">All tenants</div>
          <Button size="sm" onClick={() => setStage('creating')}>
            <Icon name="sparkles" size={13} /> New tenant
          </Button>
        </div>
        {tenants === null && (
          <div className="px-4 py-8 text-center dim">Loading…</div>
        )}
        {tenants !== null && tenants.length === 0 && (
          <div className="px-4 py-8 text-center dim">
            No tenants yet. Create one to start onboarding customers.
          </div>
        )}
        {tenants?.map((t) => (
          <TenantRow
            key={t.id}
            tenant={t}
            onOpen={() => setSelected(t.id)}
          />
        ))}
      </Card>

      {stage === 'creating' && (
        <CreateTenantSheet
          onClose={() => setStage('list')}
          onCreated={() => {
            setStage('list');
            void load();
          }}
        />
      )}

      {selected && (
        <TenantTokensSheet
          tenantId={selected}
          tenant={tenants?.find((t) => t.id === selected) ?? null}
          onClose={() => setSelected(null)}
          onDeleted={() => {
            setSelected(null);
            void load();
          }}
        />
      )}
    </div>
  );
};


// ---------------------------------------------------------------------------
// Row
// ---------------------------------------------------------------------------


const TenantRow = ({
  tenant,
  onOpen,
}: {
  tenant: TenantSummary;
  onOpen: () => void;
}) => (
  <button
    className="grid grid-cols-[1fr_auto] items-center gap-4 px-4 py-3.5 border-b border-border last:border-b-0 text-left hover:bg-muted/40 w-full"
    onClick={onOpen}
  >
    <div className="min-w-0">
      <div className="text-[13.5px] font-medium">
        {tenant.name || tenant.id}
      </div>
      <div className="dim text-xs leading-snug">
        <span className="mono">{tenant.id}</span>
        {tenant.description && ` · ${tenant.description}`}
      </div>
    </div>
    <div className="dim text-xs whitespace-nowrap">
      {formatDate(tenant.created_at)}
    </div>
  </button>
);


// ---------------------------------------------------------------------------
// Create sheet
// ---------------------------------------------------------------------------


const CreateTenantSheet = ({
  onClose,
  onCreated,
}: {
  onClose: () => void;
  onCreated: () => void;
}) => {
  const [name, setName] = useState('');
  const [slug, setSlug] = useState('');
  const [description, setDescription] = useState('');
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim()) return;
    setSaving(true);
    setError(null);
    try {
      await createTenant({
        name: name.trim(),
        slug: slug.trim() || undefined,
        description: description.trim() || undefined,
      });
      onCreated();
    } catch (e) {
      setError(
        e instanceof ApiError
          ? `${e.status}: ${e.message}`
          : `${e instanceof Error ? e.message : String(e)}`,
      );
    } finally {
      setSaving(false);
    }
  };

  return (
    <Sheet onClose={onClose}>
      <form className="flex flex-col gap-4 p-6 w-[480px]" onSubmit={submit}>
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold m-0">New tenant</h2>
          <Button size="sm" variant="ghost" onClick={onClose} type="button">
            <Icon name="x" size={14} />
          </Button>
        </div>

        <div className="flex flex-col gap-1.5">
          <label className="text-xs font-medium dim">Display name</label>
          <Input
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="Acme Corp"
            autoFocus
          />
        </div>

        <div className="flex flex-col gap-1.5">
          <label className="text-xs font-medium dim">
            Slug <span className="opacity-50">— optional, auto-derived from name</span>
          </label>
          <Input
            value={slug}
            onChange={(e) => setSlug(e.target.value)}
            placeholder="acme-corp"
            className="mono"
          />
          <div className="dim text-xs">
            Must match <code>^[a-z0-9][a-z0-9-]&#123;1,40&#125;$</code>. Used as the directory name
            under <code>tenants/</code>.
          </div>
        </div>

        <div className="flex flex-col gap-1.5">
          <label className="text-xs font-medium dim">Description (optional)</label>
          <Input
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="Production tenant for Acme"
          />
        </div>

        {error && (
          <div className="text-destructive text-xs">{error}</div>
        )}

        <div className="flex justify-end gap-2 mt-2">
          <Button variant="ghost" type="button" onClick={onClose} disabled={saving}>
            Cancel
          </Button>
          <Button type="submit" disabled={saving || !name.trim()}>
            {saving ? 'Creating…' : 'Create tenant'}
          </Button>
        </div>
      </form>
    </Sheet>
  );
};


// ---------------------------------------------------------------------------
// Tokens drill-in sheet (P16.4.S3)
// ---------------------------------------------------------------------------


const TenantTokensSheet = ({
  tenantId,
  tenant,
  onClose,
  onDeleted,
}: {
  tenantId: string;
  tenant: TenantSummary | null;
  onClose: () => void;
  onDeleted: () => void;
}) => {
  const [tokens, setTokens] = useState<TokenSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [minting, setMinting] = useState(false);
  const [mintLabel, setMintLabel] = useState('');
  const [justMinted, setJustMinted] = useState<TokenMintResponse | null>(null);
  const [confirmDelete, setConfirmDelete] = useState(false);

  const load = useCallback(async () => {
    setError(null);
    try {
      const res = await listTokens(tenantId);
      setTokens(res.tokens);
    } catch (e) {
      setError(
        e instanceof ApiError
          ? `${e.status}: ${e.message}`
          : `${e instanceof Error ? e.message : String(e)}`,
      );
    }
  }, [tenantId]);

  useEffect(() => {
    void load();
  }, [load]);

  const handleMint = async (e: React.FormEvent) => {
    e.preventDefault();
    setMinting(true);
    setError(null);
    try {
      const res = await mintToken(tenantId, mintLabel.trim());
      setJustMinted(res);
      setMintLabel('');
      await load();
    } catch (e) {
      setError(
        e instanceof ApiError
          ? `${e.status}: ${e.message}`
          : `${e instanceof Error ? e.message : String(e)}`,
      );
    } finally {
      setMinting(false);
    }
  };

  const handleRevoke = async (hashPrefix: string) => {
    try {
      await revokeToken(tenantId, hashPrefix);
      await load();
    } catch (e) {
      setError(
        e instanceof ApiError
          ? `${e.status}: ${e.message}`
          : `${e instanceof Error ? e.message : String(e)}`,
      );
    }
  };

  const handleDeleteTenant = async () => {
    try {
      await deleteTenant(tenantId);
      onDeleted();
    } catch (e) {
      setError(
        e instanceof ApiError
          ? `${e.status}: ${e.message}`
          : `${e instanceof Error ? e.message : String(e)}`,
      );
      setConfirmDelete(false);
    }
  };

  const isProtected = tenantId === '_default';

  return (
    <Sheet onClose={onClose}>
      <div className="flex flex-col gap-4 p-6 w-[560px] max-h-[90vh] overflow-auto">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold m-0">
              {tenant?.name || tenantId}
            </h2>
            <div className="dim text-xs mono">{tenantId}</div>
          </div>
          <Button size="sm" variant="ghost" onClick={onClose}>
            <Icon name="x" size={14} />
          </Button>
        </div>

        {error && (
          <Card className="border-destructive p-3">
            <p className="dim m-0 text-destructive text-xs">{error}</p>
          </Card>
        )}

        {justMinted && (
          <MintedTokenBanner
            mint={justMinted}
            onDismiss={() => setJustMinted(null)}
          />
        )}

        <Card className="gap-0 py-0">
          <div className="flex items-center justify-between border-b border-border px-4 py-3">
            <div className="text-[13px] font-semibold">Tokens</div>
            <div className="dim text-xs">
              {tokens?.length ?? 0} active
            </div>
          </div>
          <form className="flex gap-2 border-b border-border p-3" onSubmit={handleMint}>
            <Input
              value={mintLabel}
              onChange={(e) => setMintLabel(e.target.value)}
              placeholder="Label (e.g. production CLI)"
              className="flex-1"
            />
            <Button
              type="submit"
              size="sm"
              disabled={minting || !mintLabel.trim()}
            >
              {minting ? 'Minting…' : 'Mint token'}
            </Button>
          </form>
          {tokens?.length === 0 && (
            <div className="dim text-xs text-center py-6">
              No tokens yet. Mint one above.
            </div>
          )}
          {tokens?.map((t) => (
            <TokenRow
              key={t.hash_prefix}
              token={t}
              onRevoke={() => void handleRevoke(t.hash_prefix)}
            />
          ))}
        </Card>

        <div className="border-t border-border pt-4 mt-2">
          {isProtected ? (
            <div className="dim text-xs">
              The <code>_default</code> tenant cannot be deleted — it holds the bootstrap data.
            </div>
          ) : !confirmDelete ? (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setConfirmDelete(true)}
              className="text-destructive"
            >
              <Icon name="trash" size={13} /> Delete tenant
            </Button>
          ) : (
            <div className="flex items-center gap-3">
              <div className="text-xs flex-1">
                Soft-delete this tenant? Data moves to <code>tenants/_deleted/</code> and tokens are revoked.
              </div>
              <Button size="sm" variant="ghost" onClick={() => setConfirmDelete(false)}>
                Cancel
              </Button>
              <Button size="sm" onClick={handleDeleteTenant}>
                Delete
              </Button>
            </div>
          )}
        </div>
      </div>
    </Sheet>
  );
};


const TokenRow = ({
  token,
  onRevoke,
}: {
  token: TokenSummary;
  onRevoke: () => void;
}) => {
  const [confirming, setConfirming] = useState(false);
  return (
    <div className="grid grid-cols-[1fr_auto] items-center gap-3 border-b border-border last:border-b-0 px-4 py-3">
      <div className="min-w-0">
        <div className="text-[13px] font-medium">
          {token.label || <span className="dim italic">(no label)</span>}
        </div>
        <div className="dim text-xs mono">
          {token.hash_prefix}… · created {formatDate(token.created_at)}
          {token.last_used_at && ` · last used ${formatDate(token.last_used_at)}`}
        </div>
      </div>
      {confirming ? (
        <div className="flex gap-1.5">
          <Button size="sm" variant="ghost" onClick={() => setConfirming(false)}>
            Cancel
          </Button>
          <Button
            size="sm"
            onClick={() => {
              setConfirming(false);
              onRevoke();
            }}
          >
            Revoke
          </Button>
        </div>
      ) : (
        <Button
          size="sm"
          variant="ghost"
          onClick={() => setConfirming(true)}
          title="Revoke this token"
        >
          <Icon name="trash" size={13} />
        </Button>
      )}
    </div>
  );
};


const MintedTokenBanner = ({
  mint,
  onDismiss,
}: {
  mint: TokenMintResponse;
  onDismiss: () => void;
}) => {
  const [copied, setCopied] = useState(false);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(mint.token);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2000);
    } catch {
      // Old browsers without clipboard API — operator can select manually.
    }
  };

  return (
    <Card className="border-primary p-4 flex flex-col gap-3 bg-primary/5">
      <div className="flex items-start gap-2">
        <Icon name="sparkles" size={14} />
        <div className="flex-1 min-w-0">
          <div className="text-[13.5px] font-semibold">
            New token: {mint.record.label || mint.record.hash_prefix}
          </div>
          <div className="dim text-xs">
            Copy this now — it will not be shown again. Only the hash prefix
            stays on disk.
          </div>
        </div>
        <Button size="sm" variant="ghost" onClick={onDismiss}>
          <Icon name="x" size={13} />
        </Button>
      </div>
      <div className="flex gap-2 items-stretch">
        <code className="flex-1 min-w-0 overflow-x-auto bg-background border border-border rounded px-3 py-2 text-xs">
          {mint.token}
        </code>
        <Button size="sm" onClick={copy} className="shrink-0">
          {copied ? 'Copied ✓' : 'Copy'}
        </Button>
      </div>
    </Card>
  );
};


// ---------------------------------------------------------------------------
// Simple right-side sheet — we don't pull in Radix Dialog for this; the
// existing AgentSheet uses its own .sheet CSS and we mirror that.
// ---------------------------------------------------------------------------


const Sheet = ({
  onClose,
  children,
}: {
  onClose: () => void;
  children: React.ReactNode;
}) => {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [onClose]);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40"
      onClick={onClose}
    >
      <div
        className="bg-background border border-border rounded-lg shadow-lg"
        onClick={(e) => e.stopPropagation()}
      >
        {children}
      </div>
    </div>
  );
};


// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------


function formatDate(iso: string | null): string {
  if (!iso) return '—';
  try {
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return iso;
    return d.toLocaleDateString(undefined, {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    });
  } catch {
    return iso;
  }
}
