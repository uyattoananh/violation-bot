"""Print SQL for adding EXIF GPS columns to the photos table.

Run the printed SQL in the Supabase SQL editor. Idempotent.

Why these columns: phones tag every photo with GPS coordinates when
location services are on. The webapp now extracts lat/lon from EXIF on
upload (see _extract_gps in webapp/app.py) so we can render a map view
of violations without inspectors typing in coordinates by hand.

Until this migration is applied, the upload endpoint silently strips
the gps fields from the insert payload, so it's safe to deploy code
before SQL.

Usage:
  python scripts/add_exif_gps_columns.py
"""
from __future__ import annotations
import sys


def main() -> int:
    sql = """\
-- Run in Supabase SQL editor. Idempotent.

ALTER TABLE photos ADD COLUMN IF NOT EXISTS exif_lat DOUBLE PRECISION NULL;
ALTER TABLE photos ADD COLUMN IF NOT EXISTS exif_lon DOUBLE PRECISION NULL;

-- Optional but useful: index for proximity / map-bounds queries.
-- A partial index keeps it small (most photos won't have GPS).
CREATE INDEX IF NOT EXISTS idx_photos_geo
  ON photos (exif_lat, exif_lon)
  WHERE exif_lat IS NOT NULL AND exif_lon IS NOT NULL;
"""
    print(sql)
    print()
    print("Copy the SQL above into the Supabase SQL editor and run it.")
    print("Existing photos will have NULL lat/lon — only NEW uploads (and")
    print("dedup-attached re-uploads) populate the columns.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
