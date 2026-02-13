import pytest
from api.v1.routes.tests.conftest import skip_if_missing_env, new_thread_id

pytestmark = pytest.mark.integration


@skip_if_missing_env
def test_rate_limit_blocks_after_threshold(client, base_prefix):
    # assumes /qa is limited to something like "5/minute" in test env
    url = f"{base_prefix}/qa" if base_prefix else "/qa"

    for i in range(5):
        r = client.post(url, json={
            "question": "Quick test question",
            "thread_id": new_thread_id(f"rl-{i}"),
        })
        assert r.status_code == 200

    r4 = client.post(url, json={
        "question": "Quick test question",
        "thread_id": new_thread_id("rl-4"),
    })
    assert r4.status_code == 429
