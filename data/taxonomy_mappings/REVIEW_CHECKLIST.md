# Reviewer Checklist — country taxonomy mappings

This document is for the HSE professional reviewing the draft mappings
in `data/taxonomy_mappings/*.json` before they ship to production for a
new market. The mappings in this folder are currently labelled
`version: 0.1-draft` — they were built from documentation knowledge of
the destination standard, not by a credentialed reviewer.

The mapping does **not** change the model's prediction. It only
rewrites the displayed slug + label on the inspector's screen. The
canonical AECIS slug is preserved in storage and on the response so
nothing about the back-end pipeline is affected by the review.

---

## What you're reviewing

For each target standard (CSA Z1000 Canada, OSHA 29 CFR 1926 US, etc.)
there is one JSON file under `data/taxonomy_mappings/`. Each file maps:

* 29 AECIS `hse_types[*].slug` values → one local slug + label per
  entry (English required, optional French / Spanish etc.)
* 9 AECIS `locations[*].slug` values → one local slug + label per entry

The AECIS canonical slugs are listed below. Your job is to confirm
each one maps to the correct CSA / OSHA category for an inspector
who lives inside that taxonomy.

---

## How to do the review

1. Open the mapping file you're reviewing (e.g. `csa_z1000_ca.json`).
2. For each `hse_types` and `locations` entry:
   - Does the `slug` correspond to a real, recognized category under
     the target standard?
   - Is the `label_en` (and `label_fr` / etc.) appropriate to that
     standard's published vocabulary? Cite the standard reference
     where helpful (e.g. "CSA Z259.10-18" for fall arrest).
   - Are there standard references in the label that should be added
     or corrected? (current drafts include some references but not
     all).
3. Use the diagnostic preview endpoint to see a sample classification
   rewritten through your reviewed mapping:
   ```
   GET /api/taxonomies/preview?taxonomy=<id>
   ```
   You'll see top-1 + top-3 alternatives rewritten through your
   mapping. Spot-check the runner-ups too — they appear in the UI as
   one-click corrections, so they need to make sense in your country's
   vocabulary.
4. When you're satisfied, bump the file's `version` from `0.1-draft`
   to `1.0` and add yourself + your credentials to a top-level
   `reviewed_by` field, e.g.:
   ```json
   "reviewed_by": [
     {"name": "Jane Doe", "credentials": "CRSP, CSSE", "date": "2026-MM-DD"}
   ]
   ```

---

## The 29 AECIS HSE slugs (need a target mapping each)

  1. Lifting_unsafe
  2. Electrical_unsafe
  3. PPE_missing
  4. Scaffolding_unsafe
  5. Excavation_unsafe
  6. Site_general_unsafe
  7. Edge_protection_missing
  8. Chemicals_hazmat_unsafe
  9. Pressure_equipment_unsafe
 10. Fall_protection_personal
 11. Truck_vehicle_unsafe
 12. Hot_work_hazard
 13. Fire_prevention_unsafe
 14. Housekeeping_general
 15. Ladder_unsafe
 16. Site_access_unsafe
 17. Concrete_work_unsafe
 18. Mass_piling_unsafe
 19. Workshop_area_unsafe
 20. Materials_storage_unsafe
 21. Equipment_machinery_unsafe
 22. Confined_space_unsafe
 23. Formwork_unsafe
 24. First_aid_kit_unsafe
 25. Warning_signs_missing
 26. Warehouse_unsafe
 27. Common_area_unsafe
 28. Parking_area_unsafe
 29. Drinking_water_unsafe

## The 9 AECIS location slugs

  1. Common_working_area
  2. Scaffolding_and_Platform
  3. Material_area
  4. Working_at_height
  5. Electrical_equipment
  6. Lifting_work
  7. Excavation
  8. Hot_work_and_chemicals
  9. Gate_and_safety_walkway

---

## Things to watch for

### Many-to-one collisions

If the target standard lumps two AECIS classes into one (e.g. both
`Hot_work_hazard` and `Welding_unsafe` mapping to a single CSA slug),
the top-3 alternatives will appear as duplicates in the inspector UI.
Note this in the JSON's `description` field so future reviewers know
it's intentional.

### One-to-many splits

If a single AECIS class actually corresponds to multiple distinct
target categories under the destination standard (e.g. `Fall_protection_personal`
splits into "fall arrest" vs "fall restraint" under CSA Z259), pick the
single closest match for the mapping AND document the lossiness in the
file's `description` field. Future work could add a second LLM pass to
disambiguate, but for now an honest single mapping is the right call.

### Outside-scope classes

If an AECIS class genuinely has no equivalent under the target
standard, point its `slug` at the most generic catch-all in that
standard (e.g. OSHA's "general duty clause 5(a)(1)") and document the
mismatch. Don't leave the entry missing — that's how silent passthrough
bugs creep in.

### Label tone

The destination standard usually has a specific tone (e.g. OSHA prefers
regulation citations; CSA prefers descriptive names). The label_en
field should match the inspector's expectation in that country, not a
literal translation of the AECIS label.

---

## When you're done

Commit the reviewed mapping JSON on a branch named
`taxonomy-review-<country_code>` and tag the project maintainers.
A maintainer will then:

1. Run the staging eval against your reviewed mapping
2. Run the preview endpoint once more to eyeball rendering
3. Merge to `main` and deploy

Reviewer effort estimate: ~2-3 hours per country for the first review,
much less for follow-ups (typo / standard-reference fixes).
