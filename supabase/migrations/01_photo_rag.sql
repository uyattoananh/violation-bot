-- Photo-RAG migration: enable pgvector + add a reference-photo embedding table.
-- Run this once in the Supabase SQL Editor.

create extension if not exists "vector";

create table if not exists photo_embeddings (
    id              uuid primary key default gen_random_uuid(),
    -- Stable content hash so re-embeds don't duplicate
    sha256          text unique not null,
    -- Labels travel with the embedding so retrieval returns ready-to-use hints
    hse_type_slug   text,
    location_slug   text,
    label_source    text,           -- 'dtag' | 'title' | 'title_vn' | 'manual'
    project_code    text,
    issue_id        text,
    source_path     text,           -- relative path under ~/Desktop/aecis-violations (audit)
    embedding       vector(512),    -- CLIP ViT-B/32 -> 512 dim
    created_at      timestamptz not null default now()
);

-- Cosine-distance index for fast k-NN
create index if not exists photo_embeddings_cos_idx
  on photo_embeddings using ivfflat (embedding vector_cosine_ops)
  with (lists = 100);

-- Optional tenant scoping for future multi-customer deployments. Null means
-- "shared across all tenants" (our 2,423 SVN+SLPXA seed data).
alter table photo_embeddings
  add column if not exists tenant_id uuid references tenants(id) on delete set null;

create index if not exists photo_embeddings_tenant_idx
  on photo_embeddings(tenant_id);


-- Helper RPC: retrieve the k nearest neighbours to a query embedding.
-- Using cosine distance; lower = more similar.
create or replace function match_photo_embeddings(
    query_embedding vector(512),
    match_k int default 5
)
returns table (
    sha256 text,
    hse_type_slug text,
    location_slug text,
    project_code text,
    issue_id text,
    distance float
) language sql stable as $$
    select
        sha256,
        hse_type_slug,
        location_slug,
        project_code,
        issue_id,
        (embedding <=> query_embedding) as distance
    from photo_embeddings
    where embedding is not null
    order by embedding <=> query_embedding
    limit match_k;
$$;

-- Allow anyone with a valid key to read (RLS is bypassed by service_role anyway)
alter table photo_embeddings enable row level security;
drop policy if exists photo_embeddings_read on photo_embeddings;
create policy photo_embeddings_read on photo_embeddings for select using (true);
