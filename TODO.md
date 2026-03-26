2026-03-26
- [x] CDS source is not registering datetime in catalogue285,302
- [x] When downloading make sure to register sucessfull downloads in catalogue even if some fail
- [x] CDS downloads the whole year if no month is specified while CHIRPS dont (it should)
- [x] Update zarr registration logic to update catalogue without having to restart server
- [x] Update http://127.0.0.1:8000/tiles/WebMercatorQuad/map.html to stop using the old group structure
- [x] Update http://127.0.0.1:8000/tiles/WebMercatorQuad/map.html with dropdowns to select available data
- [x] Add a test command to CLI equivalent to uv run pytest tests/ --cov=eostrata --cov-report=term-missing
- [x] Add a lint command to CLI equivalent to uv run ruff check eostrata/ tests/ --fix and uv run ruff format eostrata/ tests/

2026-06-27
- [ ] CLI command to re-generate the catalogue base on available zarr data
- [ ] set default colorscale for t2m to 280,300 / coolwarm
