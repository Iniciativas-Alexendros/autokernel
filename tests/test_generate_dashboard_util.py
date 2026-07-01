import os
import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.generate_dashboard import _status_badge, _speedup_cell


class TestGenerateDashboardUtil:
    """Tests para helpers de generate_dashboard."""

    def test_status_badge_completed(self):
        html = _status_badge("completed")
        assert "completed" in html
        assert "#10b981" in html

    def test_status_badge_unknown(self):
        html = _status_badge("unknown")
        assert "unknown" in html

    def test_speedup_cell_none(self):
        html = _speedup_cell(None)
        assert "—" in html

    def test_speedup_cell_value(self):
        html = _speedup_cell(1.5)
        assert "1.500x" in html
        assert "#10b981" in html

    def test_speedup_cell_low(self):
        html = _speedup_cell(0.8)
        assert "0.800x" in html
        assert "#ef4444" in html
