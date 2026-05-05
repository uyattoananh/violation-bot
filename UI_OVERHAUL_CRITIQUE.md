# UI/UX overhaul critique — raw

Branch: `layout-rebuild-v81`. Audited at v80 on iPhone SE 375 px. Goal:
honest critique, not gentle review. Only the things I'd actually change.

---

## TL;DR — the structural problems

1. **The photos are never the dominant thing on screen.** On every surface the
   user gets to, they see a banner + a dropzone + a toolbar BEFORE the
   thing they came to look at. Inverted hierarchy.
2. **Two CTAs that do the same thing on the list view.** "+ Start new
   inspection" button AND a dashed orange "Take a photo to start a new
   inspection" tile. Pick one.
3. **Confidence is shown 3 times per photo card** (status pill text +
   numeric percent + ring colour). Pick one signal.
4. **No bulk operations.** Inspectors review 50–500 photos. They tap
   Confirm 50–500 times. There's no "select all in this category and
   confirm".
5. **Dropzone is the loudest thing on screen and is wrong 95 % of the
   time.** Inspectors upload once, review for an hour. The upload
   affordance dominates EVERY screen visit.
6. **The inspection name is buried.** It's the page identity. It should
   be the first thing your eye lands on. It's row 5.
7. **Information density is too low.** A photo card eats 380 px on a
   667-px-tall iPhone SE. You see 1.5 photos at a time. On a 50-photo
   batch that's a 19 000-px scroll.
8. **"Inspection summary" collapsible card is dead weight.** Counts
   should live in the page subtitle. Avg-confidence stat is trivia.

---

## Surface 1 — Inspection list (`/`)

![](tmp/critique/01-list.png)

### What's wrong

- **Two upload CTAs.** "+ Start new inspection" header button + the
  dashed orange "Take a photo to start a new inspection" tile below.
  Both create a new batch. The dashed tile is for users who haven't
  seen the header button — but if you've seen it once, the tile is
  noise forever.
- **Yellow 48-hour banner consumes 60 px on EVERY visit.** It's a
  one-time-realisation message ("oh, photos auto-delete") but it's
  styled as a persistent warning. Inspectors learn the rule after one
  read; the banner stays anyway.
- **Empty-state card competes with the dashed tile.** Two "no
  inspections, do something" prompts.
- **Footer "0/100" with photo icon.** Abstract. Without context
  (what's the cap? when does it reset?) it reads as a mystery counter.

### Proposed

- **Drop the dashed quick-create tile.** "+ Start new inspection" in
  the header is enough. Less code, clearer affordance.
- **Move the 48-hour notice to a tiny `info` icon** next to the
  inspection title (or to a one-line footer message). Stop dedicating
  a full row to it.
- **Empty state**: replace the big "No inspections yet." card with a
  single sentence + call-to-action right below the header. Less
  whitespace, more direct: *"No inspections yet — start your first
  with the button above ↑"* or just don't show anything at all (the
  big "+ Start new" button IS the empty state).
- **Footer quota**: only show when at >50 % usage; otherwise hide.

---

## Surface 2 — Inspection detail, empty state

![](tmp/critique/02-detail-empty.png)

### What's wrong

- **The dropzone takes 280 px** of the 667-px viewport — 42 % of the
  screen. On a fresh batch you can't see anything else.
- **Inspection name "Tap to name this inspection"** is row 5 (back
  link → expiry banner → dropzone → name). It should be row 2.
- **"Reset" button** is in the same row as the inspection name. It's
  destructive (wipes all reviews in the batch) and gets equal visual
  weight as the name.
- **Empty-state photo card** ("No photos yet") sits BELOW the
  dropzone. By the time the user scrolls to it they've already seen
  the upload affordance twice (dropzone + the empty-state's "drop
  some above"). Triple-redundant.
- **Toast "Started a new inspection"** floats over content. OK once,
  but it lands right over the inspection name field — covers the
  thing the user might want to name first.

### Proposed

- **Title row first**: editable inspection name + photo/reviewed
  counts as subtitle.
- **`[+ Add photos]` button** in title row. Tapping it expands a
  small panel below with `[Take photo]` `[Choose photos]` and the
  quota bar. Or — when batch is empty — auto-expand. Shrinks back
  when ≥1 photo exists.
- **Bury "Reset reviews" in a `[⋯]` menu.** Per-user critique: PM
  wants per-photo reset / multi-select instead of bulk wipe. Drop the
  bulk Reset; keep the inline `↶ Undo` on every reviewed card; add a
  "select multiple" mode to bulk-confirm or bulk-undo.
- **Compress the expiry banner** to one line, smaller text, an info
  icon: `[ⓘ] Expires 1d 14h · Download report` — no two-line wall
  of yellow.

---

## Surface 3 — Inspection detail, with card

![](tmp/critique/03-detail-with-card.png)

### What's wrong

- **Above the fold on iPhone SE**: back link, expiry banner,
  dropzone, name input, "Started a new inspection" toast. The actual
  photo card requires scrolling.
- **Dropzone at the top is wrong context.** User uploaded the photos
  already. They want to REVIEW them now. The dropzone is for next
  time / next batch.
- **`LIKELY CORRECT` pill + 90 % ring + emerald border + "AI
  suggests · 90 %"** — four signals saying the same thing.
- **Confidence percentage rendered twice in the same card** (in
  the ring AND in the "AI suggests · 90%" header inline).
- **`Site warning signs / barricades missing or damaged`** in bold
  AND the parent `Site warning signs missing` below in smaller
  text — when the fine label IS the parent + qualifier, the parent
  line is largely redundant. Either show one or the other.

### Proposed

- **Layout**: title row → expiry → filter+sort → photo grid. No
  dropzone above the photos when batch has photos.
- **One confidence signal**: keep the band-coloured card border (the
  card already has `band-high/medium/low`). Drop the status pill text
  ("LIKELY CORRECT") and keep just the numeric `90%`. Pill text was
  added for accessibility (color-only indicator) — replace with a
  small icon: ✓ for high, ! for medium, ? for low (still a
  non-color signal).
- **Drop the parent line** when the fine label already contains the
  parent context. Show only fine label + optional rationale.

---

## Surface 4 — Photo card edit form

![](tmp/critique/04-edit-form-open.png)

### What's wrong

- **Picker trigger looks like a search input** (icon + placeholder
  "Tap to choose or type to search"). It ISN'T a search input — tap
  opens a modal. The disguise creates two clicks: tap the input → the
  modal opens → search inside the modal. Why not skip step 1?
- **`optional note for the audit log`** is the second field, takes
  full width. We have NO idea if anyone uses this in production.
  Audit-log notes are a power-user feature buried in front of a
  primary action.
- **Save Correction** is full-width emerald, but the auxiliary
  Mark-region / Propose-new-sub-type is now behind a disclosure (good).
  However the user often just wants to pick from the alternates above
  and `Save`. Could the chips → click → save be a one-tap path? It
  isn't currently.
- **The disclosure summary "Need something else?"** uses emerald
  text as if it's a primary action. Should be slate to deemphasise.

### Proposed

- **Replace picker-trigger fake input** with a real `[Pick violation
  type ▾]` button. Tapping opens the modal (same as today). One
  cognitive step instead of two.
- **Hide the note field** behind an "Add note" link. Default-off.
  Inspectors rarely use it.
- **One-tap alts**: tap an alt-chip → it stages as the new pick AND
  the Save button label changes to "Save 'Other unsafe site
  condition'". Second tap commits. Today it stages, but the user has
  to scroll to the Save button.
- **Disclosure summary**: change from emerald to slate, italic. Less
  attention-grabbing.

---

## Surface 5 — Picker modal

![](tmp/critique/05-picker-modal.png)

### What's wrong

- **Each row is a card** with bg + border + 16 px padding. 70 px tall.
  29 categories → ~2000 px total. Too much scroll.
- **Sub-type count `2 sub-types` / `36 sub-types`** uses identical
  visual weight. The user doesn't actually care about sub-type count
  — they care about NAME.
- **No keyboard navigation** between rows (arrow keys + Enter). Mouse
  / touch only.
- **No "recently used"** even though most inspectors will return to
  the same 3-5 categories all day.
- **No grouping**. 29 flat categories with names like "Common eating
  / rest area unhygienic" next to "Excavation / pit / deep hole
  hazard" — the eye can't pre-sort. Real safety taxonomies group:
  Hazardous Substances · PPE · Equipment · Site Conditions · etc.

### Proposed

- **Tighter rows**: hairline divider, 12 px vertical padding, no
  per-row border. Roughly 44 px per row. 29 categories → 1300 px
  scroll instead of 2000 px.
- **Hide sub-type counts behind a chevron arrow** (still visible via
  the drill icon) or downsize them to `(36)` next to the name on
  the same line.
- **Keyboard nav**: ↑/↓ to move, Enter to select / drill, Esc to
  close.
- **"Recently used" section at top** of the parent list. Tracks
  user's last 5 picks in localStorage. Shortcut to the 95 % case.
- **Group categories** if the taxonomy supports it. Even adding 4–5
  optical groupings (`──── Hazards ────` `──── Site conditions ────`)
  cuts scan time in half.

---

## Cross-cutting structural issues

### A. Information density mismatch

- A photo card is 380 px tall on iPhone SE. Contains: thumbnail (224
  px) + classification block (110 px) + buttons (46 px). The
  classification block is the densest part but most of its weight is
  redundant (see Surface 3).
- **Compact card mode**: thumbnail 120 px tall, classification one
  line, ✓/✏ as icon-only buttons. Use this when the inspection has
  >20 photos. Tap-to-expand for full detail.

### B. Bulk operations are missing

- 50-photo batches are common. User taps Confirm 50 times (or worse,
  Edit → Pick → Save 50 times).
- **Multi-select mode**: long-press a card → enters multi-select →
  bulk-Confirm / bulk-Edit (apply same correction to all selected) /
  bulk-Delete.
- **Replaces global `Reset reviews`**: per the user's directive,
  Reset is per-photo (already exists as inline `↶ Undo`) plus
  multi-select reset.

### C. Onboarding / first-run

- First-time user on the list view: nothing. They have to figure out
  that "+ Start new inspection" begins the flow.
- **First-run tooltip**: pointer to the button. Or pre-create an
  inspection on first sign-in so they're already "inside".

### D. Brand consistency

- Landing page = paper-cream theme with dark ink CTAs and red-orange
  accent. Drafting / blueprint feel.
- App shell = same paper-warm bg but uses Tailwind `bg-emerald-600`
  buttons. Disconnect: the landing promises drafting/professional;
  the app delivers Tailwind-default green.
- **Pick one**: either propagate the ink+accent theme into the app
  (replace emerald with `var(--ink)` for primary buttons) OR shed the
  drafting theme on the landing in favor of standard SaaS look.

### E. Footer real estate

- Sticky footer with "Session: 0 uploaded · 0 confirmed · 0
  corrected" + quota counter + cost ticker (admin only) + "every
  confirmation trains the AI" tagline.
- **In an inspection workflow nobody cares about session totals.** The
  inspector cares about THIS BATCH'S stats, which are now in the
  title subtitle. Drop the session counters.
- **Cost ticker** is admin curiosity, not user value. Hide for
  non-admins.
- **"every confirmation trains the AI"** is marketing copy. Move to
  landing page.
- Net: footer can be a 32-px-tall strip with just the quota bar
  (when relevant) — saves 30 px on every screen.

### F. Loading / async states

- Photo upload: progress overlay covers the dropzone with a global
  "Uploading 3 / 10". No per-file progress.
- Classification: photos sit in `Analyzing…` state with a pulsing
  dot. Now has a 30-second timeout indicator (v74). Good.
- Confirm / Save / Delete: button doesn't visually change to
  "loading" — just internal `dataset.busy` flag prevents re-firing.
- **Add `disabled` + spinner glyph** on the button text during async
  operations. Today the user taps and waits silently.

### G. Sign-in flow (not screenshotted but relevant)

- Provider buttons stack: Google + Microsoft + "More options".
- Probably fine. Not a problem area.

### H. Modals overuse

- 7 modals: picker, lightbox, email export, history, markup, propose,
  kbd-help.
- Picker is mandatory (taxonomy too big for inline). Lightbox is
  appropriate. The other 5 are "secondary action" modals.
- **At least three of those could be inline-expand instead**: history
  (1 tab below the card), kbd-help (one toast on first session),
  propose (form below picker on no-results).
- Modal overuse forces user to leave context every time.

---

## Overhaul priority list

1. **Inspection detail layout rebuild** (the original PM complaint):
   - Title row at top
   - Drop hero dropzone, replace with collapsible `[+ Add]`
   - Single filter+sort row
   - No summary card
   - Save: ~280 px above-the-fold reclaimed
2. **Inspection list cleanup**:
   - Remove dashed quick-create tile (redundant CTA)
   - Compress expiry banner
   - Drop empty-state card
3. **Photo card density pass**:
   - One confidence signal (✓/!/? icon + percent)
   - Drop parent line when redundant with fine label
   - Smaller buttons; replace fake-input picker trigger with real button
4. **Multi-select mode** for bulk ops (replaces global Reset)
5. **Picker tighter rows** + recently-used + keyboard nav
6. **Footer slimming** — drop session counters and tagline
7. **Async button states** — disabled+spinner on Confirm/Save/Delete
8. **Modal-to-inline conversion** for history + propose (defer)

---

## What I'm proposing to NOT touch

- Picker modal portal architecture (recently rebuilt v53; works)
- Auth flow (works)
- Service worker / caching (recently capped v76; works)
- HEIC handling (defensive fix v75)
- Cancel/Confirm undo toast (works)
- Reduced-motion (works)
- Accessibility ARIA / contrast (verified v77/v79)

---

## Implementation plan if we ship this

Tag commits as `v81-detail-rebuild`, `v82-list-cleanup`, etc.

Estimated: 5-8 commits over the branch. Most CSS + markup; the
multi-select mode is the only new JS state machine. ~2-4 hours
focused work.

I'll wait for your sign-off on the priority order before starting.
