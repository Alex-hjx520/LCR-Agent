"""FastAPI 接口层测试（依赖项全部替换为替身）。"""

from __future__ import annotations

from api.main import API_PREFIX


class TestSystemEndpoints:
    def test_root_returns_metadata(self, api_client):
        resp = api_client.get("/")
        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == "LCR-Agent"
        assert API_PREFIX in body["api_prefix"]

    def test_health_reports_stub_llm(self, api_client):
        resp = api_client.get(f"{API_PREFIX}/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["llm_configured"] is False
        assert body["index_ready"] is False
        assert "llm" in body["details"]

    def test_ready_degrades_without_index(self, api_client):
        resp = api_client.get(f"{API_PREFIX}/ready")
        assert resp.status_code == 200
        assert resp.json()["status"] == "degraded"


class TestReviewEndpoints:
    def test_create_review_returns_report(self, api_client, sample_contract):
        resp = api_client.post(
            f"{API_PREFIX}/reviews",
            json={"text": sample_contract, "filename": "contract.txt"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "succeeded"
        assert body["report"]["filename"] == "contract.txt"
        assert body["report"]["issues"]

    def test_blank_text_is_rejected(self, api_client):
        resp = api_client.post(f"{API_PREFIX}/reviews", json={"text": "   "})
        assert resp.status_code == 422

    def test_missing_text_is_rejected(self, api_client):
        resp = api_client.post(f"{API_PREFIX}/reviews", json={"filename": "x.txt"})
        assert resp.status_code == 422

    def test_list_then_get_report(self, api_client, sample_contract):
        created = api_client.post(
            f"{API_PREFIX}/reviews", json={"text": sample_contract, "filename": "a.txt"}
        ).json()
        report_id = created["report_id"]

        listing = api_client.get(f"{API_PREFIX}/reviews").json()
        assert listing["total"] == 1
        assert listing["items"][0]["report_id"] == report_id
        assert listing["items"][0]["issue_count"] >= 1

        fetched = api_client.get(f"{API_PREFIX}/reviews/{report_id}")
        assert fetched.status_code == 200
        assert fetched.json()["report_id"] == report_id

    def test_get_unknown_report_returns_404(self, api_client):
        resp = api_client.get(f"{API_PREFIX}/reviews/rpt_not_exist")
        assert resp.status_code == 404
        assert "不存在" in resp.json()["detail"]

    def test_delete_report(self, api_client, sample_contract):
        report_id = api_client.post(
            f"{API_PREFIX}/reviews", json={"text": sample_contract}
        ).json()["report_id"]

        assert api_client.delete(f"{API_PREFIX}/reviews/{report_id}").status_code == 200
        assert api_client.get(f"{API_PREFIX}/reviews/{report_id}").status_code == 404
        assert api_client.delete(f"{API_PREFIX}/reviews/{report_id}").status_code == 404

    def test_list_pagination_params_are_validated(self, api_client):
        assert api_client.get(f"{API_PREFIX}/reviews?limit=0").status_code == 422
        assert api_client.get(f"{API_PREFIX}/reviews?limit=999").status_code == 422
        assert api_client.get(f"{API_PREFIX}/reviews?offset=-1").status_code == 422


class TestUploadEndpoint:
    def test_upload_txt_file(self, api_client, sample_contract):
        files = {"file": ("contract.txt", sample_contract.encode("utf-8"), "text/plain")}
        resp = api_client.post(
            f"{API_PREFIX}/reviews/upload", files=files, data={"use_retrieval": "false"}
        )
        assert resp.status_code == 200
        assert resp.json()["report"]["filename"] == "contract.txt"

    def test_upload_rejects_unsupported_suffix(self, api_client):
        files = {"file": ("evil.exe", b"MZ...", "application/octet-stream")}
        resp = api_client.post(f"{API_PREFIX}/reviews/upload", files=files)
        assert resp.status_code == 400
        assert "不支持的文件类型" in resp.json()["detail"]


class TestOpenApi:
    def test_schema_is_exposed(self, api_client):
        schema = api_client.get("/openapi.json").json()
        assert "/api/v1/reviews" in schema["paths"]
        assert "/api/v1/reviews/upload" in schema["paths"]
        assert schema["info"]["title"].startswith("LCR-Agent")
