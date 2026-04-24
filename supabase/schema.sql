-- Construction Violation AI — Supabase / Postgres schema
-- ----------------------------------------------------------------
-- Apply via: psql <connection-url> -f schema.sql
-- Or via Supabase dashboard: SQL Editor → paste → Run
--
-- Row-Level Security is enabled on every user-data table. The "service role"
-- key bypasses RLS for the worker backend; the "anon/authenticated" key only
-- sees rows the current user is entitled to.

create extension if not exists "pgcrypto";


-- ===============================================================
-- Tenancy & projects
-- ===============================================================

-- A PM company (tenant). One row per paying customer.
create table if not exists tenants (
    id           uuid primary key default gen_random_uuid(),
    name         text not null,
    created_at   timestamptz not null default now()
);

-- A construction project, owned by a tenant.
create table if not exists projects (
    id            uuid primary key default gen_random_uuid(),
    tenant_id     uuid not null references tenants(id) on delete cascade,
    code          text not null,    -- short label e.g. "SVN", "SLPXA"
    name          text not null,
    aecis_project_id text,          -- original id from AECIS, for cross-ref
    created_at    timestamptz not null default now(),
    unique (tenant_id, code)
);

-- App users are tied to one tenant (no multi-tenant staff for v1).
create table if not exists users (
    id         uuid primary key default gen_random_uuid(),
    tenant_id  uuid not null references tenants(id) on delete cascade,
    auth_id    uuid unique,   -- maps to Supabase auth.users.id
    email      text unique not null,
    name       text,
    role       text not null default 'inspector'
                 check (role in ('admin','inspector','viewer')),
    created_at timestamptz not null default now()
);


-- ===============================================================
-- Taxonomy (prepared vocabularies)
-- ===============================================================

-- The 19 location classes; seeded from taxonomy.json.
create table if not exists locations (
    slug      text primary key,
    label_en  text not null,
    label_vn  text
);

-- The 71 HSE-type classes; seeded from taxonomy.json.
create table if not exists hse_types (
    slug      text primary key,
    label_en  text not null,
    label_vn  text
);


-- ===============================================================
-- Photos and classifications
-- ===============================================================

-- A photo uploaded by an inspector. One row per physical photo.
create table if not exists photos (
    id               uuid primary key default gen_random_uuid(),
    tenant_id        uuid not null references tenants(id) on delete cascade,
    project_id       uuid not null references projects(id) on delete cascade,
    uploaded_by      uuid references users(id) on delete set null,
    storage_key      text not null,   -- R2/S3 object key
    storage_bucket   text not null,
    sha256           text,
    original_filename text,
    bytes            bigint,
    exif_taken_at    timestamptz,     -- from EXIF if available
    geo_lat          double precision,
    geo_lon          double precision,
    site_marker_x    double precision,  -- for "click on map" workflow
    site_marker_y    double precision,
    uploaded_at      timestamptz not null default now(),
    unique (tenant_id, sha256)
);

create index if not exists photos_project_idx on photos(project_id);
create index if not exists photos_uploaded_at_idx on photos(uploaded_at desc);


-- AI suggestion for a photo. Multiple per photo allowed (re-classify etc.).
-- `is_current = true` marks the one being shown to the inspector.
create table if not exists classifications (
    id               uuid primary key default gen_random_uuid(),
    photo_id         uuid not null references photos(id) on delete cascade,
    location_slug    text references locations(slug),
    hse_type_slug    text references hse_types(slug),
    location_confidence  real,
    hse_type_confidence  real,
    rationale        text,
    model            text,       -- e.g. "claude-sonnet-4-5-20250929"
    source           text not null default 'zero_shot'
                       check (source in ('zero_shot','fine_tuned','manual')),
    input_tokens     int,
    output_tokens    int,
    raw_response     jsonb,
    is_current       boolean not null default true,
    created_at       timestamptz not null default now()
);

create index if not exists classifications_photo_idx
  on classifications(photo_id, is_current);


-- Inspector confirmation / correction. Each correction = training-data row.
create table if not exists corrections (
    id               uuid primary key default gen_random_uuid(),
    photo_id         uuid not null references photos(id) on delete cascade,
    classification_id uuid references classifications(id) on delete set null,
    corrected_by     uuid references users(id) on delete set null,
    action           text not null
                       check (action in ('confirm','correct','reject')),
    location_slug    text references locations(slug),
    hse_type_slug    text references hse_types(slug),
    note             text,
    created_at       timestamptz not null default now()
);

create index if not exists corrections_photo_idx on corrections(photo_id);


-- Processing queue: photos waiting for AI classification.
create table if not exists classify_jobs (
    id           uuid primary key default gen_random_uuid(),
    photo_id     uuid not null references photos(id) on delete cascade,
    status       text not null default 'pending'
                   check (status in ('pending','running','done','error')),
    attempt      int not null default 0,
    error        text,
    batch_id     text,
    created_at   timestamptz not null default now(),
    updated_at   timestamptz not null default now()
);

create index if not exists classify_jobs_status_idx
  on classify_jobs(status, created_at);


-- ===============================================================
-- RLS
-- ===============================================================

alter table tenants         enable row level security;
alter table projects        enable row level security;
alter table users           enable row level security;
alter table photos          enable row level security;
alter table classifications enable row level security;
alter table corrections     enable row level security;
alter table classify_jobs   enable row level security;

-- Locations/hse_types are public read-only vocabularies.
alter table locations enable row level security;
alter table hse_types enable row level security;
drop policy if exists tax_read_loc on locations;
drop policy if exists tax_read_hse on hse_types;
create policy tax_read_loc on locations for select using (true);
create policy tax_read_hse on hse_types  for select using (true);

-- Tenant isolation: users only see rows from their own tenant.
-- Assumes JWT contains `tenant_id` custom claim (set via Supabase auth hook).
drop policy if exists tenant_isolation_projects on projects;
create policy tenant_isolation_projects on projects
  for all using (
    tenant_id = (auth.jwt() ->> 'tenant_id')::uuid
  );

drop policy if exists tenant_isolation_photos on photos;
create policy tenant_isolation_photos on photos
  for all using (
    tenant_id = (auth.jwt() ->> 'tenant_id')::uuid
  );

drop policy if exists tenant_isolation_users on users;
create policy tenant_isolation_users on users
  for all using (
    tenant_id = (auth.jwt() ->> 'tenant_id')::uuid
  );

-- Classifications / corrections / jobs inherit via photo_id.
drop policy if exists tenant_isolation_cls on classifications;
create policy tenant_isolation_cls on classifications
  for all using (
    photo_id in (
      select id from photos
      where tenant_id = (auth.jwt() ->> 'tenant_id')::uuid
    )
  );

drop policy if exists tenant_isolation_cor on corrections;
create policy tenant_isolation_cor on corrections
  for all using (
    photo_id in (
      select id from photos
      where tenant_id = (auth.jwt() ->> 'tenant_id')::uuid
    )
  );

drop policy if exists tenant_isolation_jobs on classify_jobs;
create policy tenant_isolation_jobs on classify_jobs
  for all using (
    photo_id in (
      select id from photos
      where tenant_id = (auth.jwt() ->> 'tenant_id')::uuid
    )
  );


-- ===============================================================
-- View: training_labels (inspector-confirmed ground truth)
-- ===============================================================
-- Aggregates the latest correction per photo; `final_location` and
-- `final_hse_type` are the inspector-confirmed labels ready for fine-tuning.
create or replace view training_labels as
select distinct on (p.id)
    p.id               as photo_id,
    p.tenant_id,
    p.project_id,
    p.storage_key,
    p.sha256,
    coalesce(c.location_slug, cls.location_slug) as final_location,
    coalesce(c.hse_type_slug, cls.hse_type_slug) as final_hse_type,
    c.action            as correction_action,
    c.created_at        as confirmed_at
from photos p
left join classifications cls
  on cls.photo_id = p.id and cls.is_current
left join corrections c
  on c.photo_id = p.id
order by p.id, c.created_at desc nulls last;
