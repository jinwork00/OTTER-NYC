# FloodNet NYC Urban Flood Forecasting Dataset

Code and processed data for urban flood-event extraction and flood-depth
forecasting in New York City.

The dataset combines FloodNet depth observations, IEM ASOS precipitation, NYC
catch-basin locations, and terrain features derived from the NYC 1-foot DEM.
Large data files are distributed separately from this repository.

## Setup

```bash
conda env create -f environment.yml
conda activate floodnet-nyc
```

## Data structure

Place the downloaded `data/` directory at the repository root.

```text
data/
├── 01_raw/
├── 02_preprocessed/
├── 03_qc_events/
├── 04_static/
├── 05_metadata/
└── licenses/
```

The main training table is:

```text
data/03_qc_events/annotated_grid.parquet
```

Variable definitions and file metadata are provided in
`data/05_metadata/codebook.csv` and `data/05_metadata/manifest.csv`.

## Usage

Run quality control:

```bash
python code/qc/run_pipeline.py
```

Rebuild the annotated training grid:

```bash
python code/qc/build_annotated.py --force
```

Train a baseline model:

```bash
python code/train/train.py --model mlp --loss mse --epochs 30
```

Training requires `data/04_static/sensor_static_features.parquet`, which must be
generated locally because DEM-derived products are excluded from the public
data archive.

## Data sources

- [FloodNet NYC](https://github.com/floodnet-nyc/floodnet-data)
- [IEM ASOS](https://mesonet.agron.iastate.edu/request/asos/1min.phtml)
- [NYCDEP Citywide Catch Basins](https://data.cityofnewyork.us/Environment/NYCDEP-Citywide-Catch-Basins/2w2g-fk3i)
- [NYC 1-foot DEM](https://data.cityofnewyork.us/City-Government/1-foot-Digital-Elevation-Model-DEM-/dpc8-z3jc/about_data)

## License

Project-created data contributions are provided under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), subject
to upstream terms. See [`data/licenses/`](data/licenses/).

## Data access

**Dataset download:** Link will be added when the dataset is published.
