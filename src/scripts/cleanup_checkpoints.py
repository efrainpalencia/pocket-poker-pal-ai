import os
import sys
from datetime import datetime, timedelta, timezone

import psycopg

# ENV VARIABLES
# DATABASE_URL (required)
# CHECKPOINT_RETENTION_HOURS (default 24)

DATABASE_URL = os.getenv("DATABASE_URL")
RETENTION_HOURS = int(os.getenv("CHECKPOINT_RETENTION_HOURS", "24"))

if not DATABASE_URL:
    print("DATABASE_URL not set")
    sys.exit(1)


def main():
    cutoff = datetime.now(timezone.utc) - timedelta(hours=RETENTION_HOURS)

    print(f"Deleting checkpoints older than {cutoff.isoformat()}")

    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            # Delete from main checkpoints table
            cur.execute(
                """
                DELETE FROM checkpoints
                WHERE created_at < %s
                """,
                (cutoff,),
            )
            deleted = cur.rowcount
            print(f"Deleted {deleted} checkpoint rows")

        conn.commit()

    print("Cleanup complete.")


if __name__ == "__main__":
    main()
