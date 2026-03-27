2026-03-26 - Afternoon
- [x] CDS source is not registering datetime in catalogue285,302
- [x] When downloading make sure to register sucessfull downloads in catalogue even if some fail
- [x] CDS downloads the whole year if no month is specified while CHIRPS dont (it should)
- [x] Update zarr registration logic to update catalogue without having to restart server
- [x] Update http://127.0.0.1:8000/tiles/WebMercatorQuad/map.html to stop using the old group structure
- [x] Update http://127.0.0.1:8000/tiles/WebMercatorQuad/map.html with dropdowns to select available data
- [x] Add a test command to CLI equivalent to uv run pytest tests/ --cov=eostrata --cov-report=term-missing
- [x] Add a lint command to CLI equivalent to uv run ruff check eostrata/ tests/ --fix and uv run ruff format eostrata/ tests/

2026-03-27 - Morning
- [x] CLI / API command to re-generate the catalogue base on available zarr data
- [x] Add a tab to eostrata viewer to demo the new ingest endpoints, including jobs tracking
- [x] Display eostrata config bounding box in viewer
- [x] Make the tab layout in viewer look more like a tab
- [x] Add config tab in viewer which displays eostrata config details (boundingbox, quota)
- [x] Make sure the API is OCG compliant
- [x] Consistent and OCG compliant API error hanling

2026-03-27 - Afternoon
- [x] Review eviction strategy:
  - [x] Allow setting an optional buffer in MB in configuration, comment can recommend 10% of quota, if buffer superior to quota don't use it during eviction
  - [x] It must be possible to evict data base on specific timestamp (not entire groups). Make recording of last access (via tiles or zonalstats) an optional configuration setting (otherwise last access = ingestion timestamp)
- [ ] Sources / ingestion
  - [ ] Support a daily data source (eg ERA5-land)
  - [x] Support key word ALL = 1,2,3,4,5,6,7,8,9,10,11,12 for months param
  - [ ] Support key word ALL = 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31 for days param

Later
- [ ] Add UI to /map for the scheduler (a new tab)
- [ ] Think of a way to encapsulate sources so it is easy to plug new ones, eg a single .py file in the source folder and you get the cli command the api and so, propose a template so anyone can use it as a model to add new sources.
