"""Smoke-test R2 credentials.

Reads creds from .env, runs three operations against the bucket:
  1. head_bucket           — confirm creds + bucket name are valid
  2. put_object (tiny)     — write permission
  3. get_object            — read permission + presign round-trip
  4. delete_object         — cleanup (don't leave test file behind)

Exits 0 on success. Prints every step so you can see what passed/failed.

Usage:
  python scripts/test_r2.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass


def main() -> int:
    account_id = os.environ.get("R2_ACCOUNT_ID")
    ak = os.environ.get("R2_ACCESS_KEY_ID")
    sk = os.environ.get("R2_SECRET_ACCESS_KEY")
    bucket = os.environ.get("R2_BUCKET", "aecis-violations")

    missing = [n for n, v in [
        ("R2_ACCOUNT_ID", account_id),
        ("R2_ACCESS_KEY_ID", ak),
        ("R2_SECRET_ACCESS_KEY", sk),
    ] if not v or v == "REPLACE_ME"]
    if missing:
        print(f"Missing env vars: {', '.join(missing)}", file=sys.stderr)
        return 2

    import boto3
    from botocore.exceptions import ClientError

    s3 = boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=ak,
        aws_secret_access_key=sk,
        region_name="auto",
    )

    test_key = f"_r2_test_{os.getpid()}.txt"
    test_body = b"hello r2"

    try:
        s3.head_bucket(Bucket=bucket)
        print(f"[1/4] head_bucket({bucket!r})                 OK")
    except ClientError as e:
        print(f"[1/4] head_bucket failed: {e}", file=sys.stderr)
        return 1

    try:
        s3.put_object(Bucket=bucket, Key=test_key, Body=test_body, ContentType="text/plain")
        print(f"[2/4] put_object(key={test_key!r})           OK")
    except ClientError as e:
        print(f"[2/4] put_object failed: {e}", file=sys.stderr)
        return 1

    try:
        obj = s3.get_object(Bucket=bucket, Key=test_key)
        body = obj["Body"].read()
        assert body == test_body
        print(f"[3/4] get_object round-trip                    OK  ({len(body)} bytes)")
    except Exception as e:  # noqa: BLE001
        print(f"[3/4] get_object failed: {e}", file=sys.stderr)
        return 1

    try:
        presigned = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": test_key},
            ExpiresIn=60,
        )
        print(f"[3b] presigned URL                              OK")
        print(f"     {presigned[:100]}...")
    except ClientError as e:
        print(f"[3b] presign failed: {e}", file=sys.stderr)
        return 1

    try:
        s3.delete_object(Bucket=bucket, Key=test_key)
        print(f"[4/4] delete_object                             OK")
    except ClientError as e:
        print(f"[4/4] delete_object failed: {e}", file=sys.stderr)
        return 1

    print()
    print("R2 OK — credentials + bucket are ready for the webapp.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
