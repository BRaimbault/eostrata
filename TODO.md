2026-03-26
- [x] CDS source is not registering datetime in catalogue285,302
- [x] When downloading make sure to register sucessfull downloads in catalogue even if some fail
- [x] CDS downloads the whole year if no month is specified while CHIRPS dont (it should)
- [x] Update zarr registration logic to update catalogue without having to restart server
- [x] Update http://127.0.0.1:8000/tiles/WebMercatorQuad/map.html to stop using the old group structure
- [x] Update http://127.0.0.1:8000/tiles/WebMercatorQuad/map.html with dropdowns to select available data
- [x] Add a test command to CLI equivalent to uv run pytest tests/ --cov=eostrata --cov-report=term-missing
- [x] Add a lint command to CLI equivalent to uv run ruff check eostrata/ tests/ --fix and uv run ruff format eostrata/ tests/

2026-03-27
- [x] CLI / API command to re-generate the catalogue base on available zarr data
- [x] Add a tab to eostrata viewer to demo the new ingest endpoints, including jobs tracking
- [x] Display eostrata config bounding box in viewer
- [x] Make the tab layout in viewer look more like a tab
- [x] Add config tab in viewer which displays eostrata config details (boundingbox, quota)
- [x] Make sure the API is OCG compliant
- [x] Consistent and OCG compliant API error hanling
- [ ] Think of a way to encapsulate sources so it is easy to plug new ones, eg a single .py file in the source folder and you get the cli command the api and so, propose a template so anyone can use it as a model to add new sources.
