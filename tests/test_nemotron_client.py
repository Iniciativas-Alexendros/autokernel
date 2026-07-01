import subprocess
import sys
from unittest import mock

import pytest

sys.path.insert(0, "/home/alexendros/repositorios/org-iniciativas-alexendros/autokernel")

from autokernel.nemotron_client import NemotronClient, ReviewResult, _read_proton_pass


class TestNemotronClient:
    """Tests para NemotronClient."""

    def test_read_proton_pass_success(self):
        with mock.patch("autokernel.nemotron_client.subprocess.run") as fake_run:
            fake_run.return_value = mock.Mock(stdout="secret-key\n", returncode=0)
            key = _read_proton_pass("Infraestructura", "NVIDIA", "APIkey")
            assert key == "secret-key"
            fake_run.assert_called_once()

    def test_read_proton_pass_failure(self):
        with (
            mock.patch("autokernel.nemotron_client.subprocess.run") as fake_run,
            mock.patch.dict("os.environ", {}, clear=True),
        ):
            fake_run.side_effect = subprocess.CalledProcessError(1, "pass-cli", stderr="not found")
            with pytest.raises(RuntimeError):
                _read_proton_pass("Infraestructura", "NVIDIA", "APIkey")

    def test_review_result_parse_approved(self):
        raw = "Approved\n- no issues"
        result = ReviewResult.parse(raw)
        assert result.approved is True

    def test_review_result_parse_rejected(self):
        raw = "Critical issue\n- memory bug"
        result = ReviewResult.parse(raw)
        assert result.approved is False
        assert "memory bug" in result.critical_issues

    def test_client_init(self):
        with mock.patch("autokernel.nemotron_client._read_proton_pass", return_value="fake-key"):
            client = NemotronClient()
            assert client.api_key == "fake-key"
