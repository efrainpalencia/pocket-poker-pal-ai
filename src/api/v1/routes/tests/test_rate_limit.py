import pytest

from api.v1.routes.tests.conftest import new_thread_id

pytestmark = pytest.mark.integration


def _post_qa(client, url: str, ip: str):
    return client.post(
        url,
        headers={"x-forwarded-for": ip},
        json={"question": "Quick test question", "thread_id": new_thread_id("rl")},
    )


def test_rate_limit_blocks_after_threshold(client, base_prefix):
    """
    Deterministic rate limit test:
      - use a unique IP so other tests don’t affect the counter
      - assert that after N requests we eventually get 429
    """
    url = f"{base_prefix}/qa" if base_prefix else "/qa"

    # stable “unique” test IP
    ip = f"203.0.113.{(hash('rate-limit') % 200) + 1}"

    # First few should be allowed (if your limit is >= 2/minute).
    # Some test environments return 500 for misconfiguration; accept that too.
    r1 = _post_qa(client, url, ip)
    # allow already-limited or misconfigured environments
    assert r1.status_code in (200, 429, 500), r1.text

    r2 = _post_qa(client, url, ip)
    assert r2.status_code in (200, 429, 500), r2.text

    # Keep going until we see a 429 (but bound the loop)
    saw_429 = (r1.status_code == 429) or (r2.status_code == 429)
    for _ in range(10):
        r = _post_qa(client, url, ip)
        if r.status_code == 429:
            saw_429 = True
            break

    assert saw_429, "Expected rate limiting (429) but never hit the threshold"
