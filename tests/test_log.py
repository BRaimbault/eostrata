"""Tests for eostrata/log.py — logging setup."""

from __future__ import annotations

import logging
import logging.handlers
from unittest.mock import MagicMock, patch


def _run_setup_logging(**kwargs):
    """Run setup_logging against a fresh mock root logger, bypassing the duplicate guard."""
    mock_root = MagicMock()
    mock_root.handlers = []  # empty → won't return early
    with patch("eostrata.log.logging.getLogger", return_value=mock_root):
        from eostrata.log import setup_logging

        setup_logging(**kwargs)
    return mock_root


class TestSetupLogging:
    def test_verbose_sets_debug_level(self):
        root = _run_setup_logging(verbose=True, log_file="", rich_console=False)
        root.setLevel.assert_called_with(logging.DEBUG)

    def test_non_verbose_sets_info_level(self):
        root = _run_setup_logging(verbose=False, log_file="", rich_console=False)
        root.setLevel.assert_called_with(logging.INFO)

    def test_duplicate_guard_returns_early(self):
        """If root already has handlers, setup_logging must not add more."""
        mock_root = MagicMock()
        mock_root.handlers = [MagicMock()]  # non-empty → early return
        with patch("eostrata.log.logging.getLogger", return_value=mock_root):
            from eostrata.log import setup_logging

            setup_logging(log_file="", rich_console=False)
        mock_root.addHandler.assert_not_called()

    def test_rich_console_handler_added(self):
        root = _run_setup_logging(log_file="", rich_console=True)
        added_types = [type(call.args[0]).__name__ for call in root.addHandler.call_args_list]
        assert "RichHandler" in added_types

    def test_no_rich_console_when_disabled(self):
        root = _run_setup_logging(log_file="", rich_console=False)
        added_types = [type(call.args[0]).__name__ for call in root.addHandler.call_args_list]
        assert "RichHandler" not in added_types
        # A plain StreamHandler must still be added so app logs reach the console
        assert "StreamHandler" in added_types

    def test_file_handler_added(self, tmp_path):
        root = _run_setup_logging(log_file=str(tmp_path / "test.log"), rich_console=False)
        added_types = [type(call.args[0]).__name__ for call in root.addHandler.call_args_list]
        assert "TimedRotatingFileHandler" in added_types
        assert (tmp_path).exists()

    def test_empty_log_file_disables_file_handler(self):
        root = _run_setup_logging(log_file="", rich_console=False)
        added_types = [type(call.args[0]).__name__ for call in root.addHandler.call_args_list]
        assert "TimedRotatingFileHandler" not in added_types

    def test_none_log_file_uses_settings(self, tmp_path):
        mock_settings = MagicMock()
        mock_settings.log_file = str(tmp_path / "settings.log")
        with patch("eostrata.config.settings", mock_settings):
            root = _run_setup_logging(log_file=None, rich_console=False)
        added_types = [type(call.args[0]).__name__ for call in root.addHandler.call_args_list]
        assert "TimedRotatingFileHandler" in added_types

    def test_creates_parent_dirs_for_log_file(self, tmp_path):
        log_path = tmp_path / "nested" / "deep" / "eostrata.log"
        _run_setup_logging(log_file=str(log_path), rich_console=False)
        assert log_path.parent.exists()


class TestSuppressPollingFilter:
    def test_allows_non_polling_requests(self):
        from eostrata.log import _SuppressPollingFilter

        f = _SuppressPollingFilter()
        record = __import__("logging").makeLogRecord(
            {"msg": 'GET /collections/worldpop HTTP/1.1" 200'}
        )
        assert f.filter(record) is True

    def test_suppresses_jobs_endpoint(self):
        from eostrata.log import _SuppressPollingFilter

        f = _SuppressPollingFilter()
        record = __import__("logging").makeLogRecord({"msg": 'GET /processes/jobs HTTP/1.1" 200'})
        assert f.filter(record) is False

    def test_suppresses_store_usage_endpoint(self):
        from eostrata.log import _SuppressPollingFilter

        f = _SuppressPollingFilter()
        record = __import__("logging").makeLogRecord({"msg": 'GET /store-usage HTTP/1.1" 200'})
        assert f.filter(record) is False
