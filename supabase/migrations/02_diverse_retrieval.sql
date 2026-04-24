-- Class-diverse retrieval: fix the housekeeping-bias in the original RPC.
-- Original returned the k globally-nearest photos. When Housekeeping has 475
-- training photos and Scaffolding has 111, the top-5 nearest are almost
-- always 4-5 housekeeping even for clearly-scaffolding scenes.
--
-- New strategy: global top-3 by distance + up to (k - 3) diverse picks,
-- where "diverse" means "best-in-class for classes not already in the
-- result set". Signs Sonnet that rare classes also look close, so the model
-- doesn't default to the dominant class.
--
-- Run this once in the Supabase SQL editor. Safe to re-run.

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
    with ranked as (
        select
            pe.sha256,
            pe.hse_type_slug,
            pe.location_slug,
            pe.project_code,
            pe.issue_id,
            (pe.embedding <=> query_embedding) as distance,
            row_number() over (
                partition by pe.hse_type_slug
                order by pe.embedding <=> query_embedding
            ) as rnk_in_class
        from photo_embeddings pe
        where pe.embedding is not null
    ),
    top_overall as (
        -- 60% of budget: absolute-nearest (any class)
        select sha256, hse_type_slug, location_slug, project_code, issue_id,
               distance, 1 as priority
        from ranked
        order by distance
        limit greatest(1, (match_k * 3 / 5))
    ),
    diverse as (
        -- 40%: best-in-class for classes NOT in top_overall
        select sha256, hse_type_slug, location_slug, project_code, issue_id,
               distance, 2 as priority
        from ranked
        where rnk_in_class = 1
          and hse_type_slug not in (select hse_type_slug from top_overall)
        order by distance
        limit greatest(1, match_k - (match_k * 3 / 5))
    ),
    combined as (
        select * from top_overall
        union all
        select * from diverse
    )
    select sha256, hse_type_slug, location_slug, project_code, issue_id, distance
    from combined
    order by priority, distance
    limit match_k;
$$;
