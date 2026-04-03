"""Tests for the CDS/ERA5 source."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from eostrata.constants import PROP_VARIABLE
from eostrata.sources.cds import _VARIABLE_MAP, CDSSource


class TestVariableMap:
    def test_known_variables(self):
        assert "t2m" in _VARIABLE_MAP
        assert "tp" in _VARIABLE_MAP
        assert _VARIABLE_MAP["t2m"] == "2m_temperature"
        assert _VARIABLE_MAP["tp"] == "total_precipitation"


class TestCDSSource:
    def setup_method(self):
        self.source = CDSSource()

    def test_metadata(self):
        assert self.source.id == "cds"
        assert self.source.collection_id == "era5"
        assert self.source.temporal_resolution == "monthly"

    def test_zarr_group_default(self):
        assert self.source.zarr_group() == "era5/t2m"

    def test_zarr_group_custom(self):
        assert self.source.zarr_group(variable="tp") == "era5/tp"

    def test_stac_item_id(self):
        assert self.source.stac_item_id() == "t2m"
        assert self.source.stac_item_id(variable="tp") == "tp"

    def test_stac_properties(self):
        props = self.source.stac_properties(variable="t2m", year=2020)
        assert props[PROP_VARIABLE] == "t2m"
        assert props["eostrata:cds_variable"] == "2m_temperature"

    def test_latest_available_is_in_past(self):
        latest = self.source.latest_available()
        assert latest < datetime.now(tz=UTC)

    def test_latest_available_has_timezone(self):
        latest = self.source.latest_available()
        assert latest.tzinfo is not None

    def test_latest_available_january_wraps_to_previous_year(self, mocker):
        """In January, subtracting 3 months wraps to October of the previous year."""
        mock_dt = mocker.patch("eostrata.sources.cds.datetime")
        mock_dt.now.return_value = datetime(2024, 1, 15, tzinfo=UTC)
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        latest = CDSSource().latest_available()

        assert latest == datetime(2023, 10, 1, tzinfo=UTC)

    def test_cdsapi_import_error_message(self, mocker):
        """Helpful error message shown when cdsapi is not installed."""
        import sys

        mocker.patch.dict(sys.modules, {"cdsapi": None})
        from eostrata.sources.cds import _get_cdsapi

        with pytest.raises(ImportError, match="cdsapi"):
            _get_cdsapi()

    def test_cdsapi_success_returns_module(self, mocker):
        """_get_cdsapi returns the cdsapi module when it is available."""
        import sys

        fake_cdsapi = mocker.MagicMock()
        mocker.patch.dict(sys.modules, {"cdsapi": fake_cdsapi})
        from eostrata.sources.cds import _get_cdsapi

        result = _get_cdsapi()
        assert result is fake_cdsapi


class TestDownloadEra5:
    def test_skips_existing_file(self, tmp_path):
        """_download_era5 returns immediately when the destination file exists."""
        from eostrata.sources.cds import _download_era5

        dest = tmp_path / "era5_t2m_2023.nc"
        dest.write_bytes(b"existing")
        result = _download_era5(
            dest, variable="2m_temperature", year=2023, months=[1], bbox=(0, 0, 10, 10)
        )
        assert result == dest

    def test_calls_cdsapi_client(self, tmp_path, mocker):
        """_download_era5 calls cdsapi.Client.retrieve when file is missing."""
        import sys

        from eostrata.sources.cds import _download_era5

        dest = tmp_path / "era5_t2m_2023.nc"

        fake_client = mocker.MagicMock()
        fake_cdsapi = mocker.MagicMock()
        fake_cdsapi.Client.return_value = fake_client

        def _fake_retrieve(_dataset, _params, path):
            Path(path).write_bytes(b"nc content")

        fake_client.retrieve.side_effect = _fake_retrieve

        mocker.patch.dict(sys.modules, {"cdsapi": fake_cdsapi})
        mocker.patch("eostrata.sources.cds._get_cdsapi", return_value=fake_cdsapi)
        result = _download_era5(
            dest, variable="2m_temperature", year=2023, months=[1, 2], bbox=(2, 4, 15, 14)
        )

        assert result == dest
        fake_client.retrieve.assert_called_once()
