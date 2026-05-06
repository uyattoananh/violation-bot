/* HSE Detector — Service Worker
 *
 * Two cache strategies:
 *
 *   1. APP SHELL (precache): the static HTML/CSS/JS + icons get
 *      cached on install. Cache-first on fetch — instant load,
 *      works offline. Bumped on every meaningful release via the
 *      CACHE_VERSION below.
 *
 *   2. RUNTIME: photo thumbnails (R2 presigned URLs), JSON API
 *      responses. Network-first with a stale-while-revalidate
 *      fallback for thumbnails — fresh data when online, last-known
 *      thumbnail when offline.
 *
 * What we DELIBERATELY do NOT cache:
 *   - /api/upload  (must be online to classify; queueing is a future
 *                   addition via background sync, not v1)
 *   - /api/usage/today, /api/pending, /api/batches  (data freshness
 *                   matters more than offline access)
 *   - /api/photos/{id}/{retry,history,reclassify-region}  (mutations)
 *
 * Increment CACHE_VERSION whenever the app shell changes (e.g. new JS
 * functions in index.html). Old caches get cleaned up on activate.
 */

// Bump on every release that changes the rendered HTML/JS. The
// browser fetches the SW file on each navigation; a different
// version string triggers re-precache + cleanup of the old
// cache name in `activate`. Phase 5 added the auth-pill markup
// + JS — without this bump, returning users keep seeing the
// pre-Phase-5 shell.
const CACHE_VERSION = "v111-list-hairline-rows";
const SHELL_CACHE = `violation-ai-shell-${CACHE_VERSION}`;
const RUNTIME_CACHE = `violation-ai-runtime-${CACHE_VERSION}`;

// Cap the runtime cache so a long-running PWA install doesn't fill the
// browser's quota and start failing every subsequent cache.put silently.
// 200 entries × ~150 KB R2 thumbnails ≈ 30 MB, well under typical PWA
// quotas (60-100 MB+). The cache holds R2 thumbnails (bulk),
// /auth/me, /api/batches, /api/usage/today (each <5 KB, negligible).
// UX backlog 8.4. FIFO not strict-LRU because the Cache API doesn't
// expose access time and re-putting on every read would double traffic;
// FIFO matches user expectation here (oldest thumbnails are from
// inactive inspections — safe to evict first).
const RUNTIME_MAX_ENTRIES = 200;

// App-shell URLs to precache. Keep this minimal — anything that
// changes per-deploy needs to be in here so the bumped CACHE_VERSION
// triggers a refetch.
const SHELL_URLS = [
  "/",
  "/static/app.css",
  "/static/manifest.json",
  "/static/icons/icon-192.png",
  "/static/icons/icon-512.png",
  "/static/icons/icon-180.png",
];

// =============================================================
// install: precache the app shell
// =============================================================
self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(SHELL_CACHE).then((cache) => {
      return cache.addAll(SHELL_URLS).catch((err) => {
        // If any single resource fails (e.g. icon not yet generated)
        // don't abort the whole install — log + continue.
        console.warn("[sw] partial precache:", err);
      });
    })
  );
  // Take effect on the very next page load instead of waiting for
  // all open tabs to close.
  self.skipWaiting();
});

// =============================================================
// activate: clean up stale caches from old versions
// =============================================================
self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((names) => {
      return Promise.all(
        names
          .filter((n) => n !== SHELL_CACHE && n !== RUNTIME_CACHE)
          .map((n) => caches.delete(n))
      );
    }).then(() => self.clients.claim())
  );
});

// =============================================================
// fetch: route requests through the right strategy
// =============================================================
self.addEventListener("fetch", (event) => {
  const req = event.request;
  const url = new URL(req.url);

  // Only handle GET. Let everything else pass through (uploads,
  // confirms, corrections, retries, mark-region).
  if (req.method !== "GET") return;

  // Same-origin only — never proxy R2 / OpenRouter / Supabase.
  // (R2 thumbnails come from a different origin and are handled
  // separately below.)
  const isSameOrigin = url.origin === self.location.origin;

  // R2 thumbnails (cross-origin) — stale-while-revalidate
  // Detect by the typical R2 URL shape: *.r2.cloudflarestorage.com
  if (url.hostname.endsWith(".r2.cloudflarestorage.com")) {
    event.respondWith(_staleWhileRevalidate(req));
    return;
  }

  // /auth/me — network-first so PWA offline opens show the last-known
  // auth state (avatar+name) instead of flashing "Sign in".
  if (isSameOrigin && url.pathname === "/auth/me") {
    event.respondWith(_networkFirst(req));
    return;
  }

  // Same-origin API JSON — network-first, fall back to cache only
  // for known idempotent endpoints (taxonomy, batches LIST). Skip
  // mutating endpoints (POST is filtered above).
  if (isSameOrigin && url.pathname.startsWith("/api/")) {
    // Only cache the read-only listing endpoints. Photo-specific
    // ones change too often to be useful from cache.
    if (
      url.pathname === "/api/batches" ||
      url.pathname === "/api/usage/today"
    ) {
      event.respondWith(_networkFirst(req));
      return;
    }
    // Everything else under /api/ goes straight to network. If the
    // user is offline, it just fails — same as today.
    return;
  }

  // The bare `/` is auth-dependent: it renders the landing page for
  // anonymous visitors and the app shell for authenticated ones.
  // Cache-first would freeze whichever copy was cached first, so a
  // user who signed out and refreshed would still see the app shell
  // (or vice versa). Network-first means each load asks the server
  // which page to render and the cache is only an offline fallback.
  if (isSameOrigin && url.pathname === "/") {
    event.respondWith(_networkFirst(req));
    return;
  }

  // Static assets — cache-first (they're versioned + immutable in
  // intent; SW evicts old caches on activate when CACHE_VERSION bumps).
  if (isSameOrigin && url.pathname.startsWith("/static/")) {
    event.respondWith(_cacheFirst(req));
    return;
  }
});

// =============================================================
// cache helpers — bounded put + FIFO prune
// =============================================================

// Wrap cache.put so quota-exceeded failures don't bubble up as
// unhandled rejections. When the browser quota is full, put() rejects
// with a QuotaExceededError; we want to swallow that and let the next
// request just go to network. The follow-up _pruneRuntimeCache call
// will free room for the next put.
async function _safePut(cache, req, resp) {
  try {
    await cache.put(req, resp);
  } catch (err) {
    // Don't log every failure (storms in offline-then-online sessions);
    // log first failure per session to surface the issue once.
    if (!self._cachePutFailedLogged) {
      console.warn("[sw] cache.put failed (quota?):", err);
      self._cachePutFailedLogged = true;
    }
  }
}

// FIFO eviction: keep RUNTIME_MAX_ENTRIES; delete the oldest first.
// Cache API preserves insertion order in keys(). Fire-and-forget from
// the strategy handlers — pruning is best-effort and shouldn't block
// the response.
async function _pruneRuntimeCache() {
  try {
    const cache = await caches.open(RUNTIME_CACHE);
    const keys = await cache.keys();
    const overflow = keys.length - RUNTIME_MAX_ENTRIES;
    if (overflow <= 0) return;
    for (let i = 0; i < overflow; i++) {
      await cache.delete(keys[i]);
    }
  } catch (err) {
    console.warn("[sw] runtime cache prune failed:", err);
  }
}

// =============================================================
// strategy implementations
// =============================================================

async function _cacheFirst(req) {
  const cache = await caches.open(SHELL_CACHE);
  const cached = await cache.match(req);
  if (cached) return cached;
  try {
    const resp = await fetch(req);
    if (resp.ok) await _safePut(cache, req, resp.clone());
    return resp;
  } catch (err) {
    // Offline + not in cache — return a minimal offline page if it
    // was a navigation request, otherwise let the fetch error
    // propagate.
    if (req.mode === "navigate") {
      const fallback = await cache.match("/");
      if (fallback) return fallback;
    }
    throw err;
  }
}

async function _networkFirst(req) {
  const cache = await caches.open(RUNTIME_CACHE);
  try {
    const resp = await fetch(req);
    if (resp.ok) {
      await _safePut(cache, req, resp.clone());
      _pruneRuntimeCache();   // fire-and-forget
    }
    return resp;
  } catch (err) {
    const cached = await cache.match(req);
    if (cached) return cached;
    throw err;
  }
}

async function _staleWhileRevalidate(req) {
  const cache = await caches.open(RUNTIME_CACHE);
  const cached = await cache.match(req);
  // Always kick off a fresh fetch in the background to update the
  // cache, so next time the photo loads with fresh data even if the
  // R2 presigned URL is about to expire.
  const fetchAndUpdate = fetch(req)
    .then(async (resp) => {
      if (resp.ok) {
        await _safePut(cache, req, resp.clone());
        _pruneRuntimeCache();
      }
      return resp;
    })
    .catch(() => null);
  return cached || (await fetchAndUpdate) || new Response(null, { status: 504 });
}

// =============================================================
// message: allow page to ask SW to skipWaiting (future use, e.g.
// "update available — reload?" UI)
// =============================================================
self.addEventListener("message", (event) => {
  if (event.data === "SKIP_WAITING") self.skipWaiting();
});
