# Sim-Equity: Agent-Based Simulation of Equitable Emergency Evacuation in Île-de-France

[![NetMob25](https://img.shields.io/badge/NetMob25-Data%20Challenge-blue)](https://netmob.org/www25/)
<!--[![Zenodo](https://img.shields.io/badge/Zenodo-17074045-blue)](https://zenodo.org/records/17074045)-->
[![Python](https://img.shields.io/badge/Python-3.12-green)](https://python.org)
<!--[![License](https://img.shields.io/badge/License-MIT-lightgrey)](#license)-->

> **Official companion repository** for the paper:
> *A Simulation Study on Equitable Mobility During City Emergencies, Focusing on Vulnerable Groups*
> Youssef M. Abd ElHameid¹ · Noha Gamal ElDin²
> ¹ School of Computational Sciences & AI, Zewail City of Science and Technology
> ² Computer Science Program, Nile University

---

## Overview

Traditional urban evacuation models treat populations as homogeneous flows routed
through a network — an approach that is dangerously incomplete. The capacity to
respond to a crisis is not uniform: a person's age, physical ability, household
structure, and socioeconomic status fundamentally determine their ability to
evacuate before those around them are already safe.

This repository implements a two-stage computational framework to **quantify the
Evacuation Equity Gap** — the measurable difference in evacuation outcomes
between the most and least socially vulnerable individuals — during a large-scale
crisis simulation in the Greater Paris (Île-de-France) region.

### Research questions

1. Can a robust, multi-dimensional **Social Vulnerability Index (SVI)** be
   constructed from individual-level mobility and sociodemographic data,
   moving beyond broad "special needs" categories?
2. Is there a quantifiable **Evacuation Equity Gap** between vulnerability groups
   in terms of success rate and evacuation time?
3. How do vulnerability-driven behaviors interact with urban mobility modes
   (walking, cycling, motorized vehicles, public transit) to produce emergent
   system-level outcomes such as gridlock and widespread evacuation failure?

### Key findings

| Finding                     | Detail                                                                                           |
|-----------------------------|--------------------------------------------------------------------------------------------------|
| Overall success rate        | **18.3%** evacuated within 3 hours — extreme difficulty of urban mass evacuation                 |
| Equity Gap is non-monotonic | U-shaped relationship between SVI and outcomes, driven by mode × congestion interaction          |
| Failure by gridlock         | MPV-heavy groups trapped in congestion despite vehicle access                                    |
| Failure by isolation        | Pedestrian groups unable to cover sufficient distance on foot                                    |
| Paradox of private vehicles | Car access — a peacetime resilience asset — becomes a systemic liability during mass evacuations |
| Dominant observed mode      | Walking + public transit (57.6% of agents)                                                       |

---

## Methodology

The framework operates in two sequential stages:

### Stage 1 — Social Vulnerability Index (SVI) Construction

Individual-level SVI scores are derived from 14 sociodemographic variables in
the NetMob25 dataset:

- **Demographic**: `SEX`, `AGE`, `DIPLOMA`, `PMR` (reduced mobility status)
- **Household**: `NBPERS_HOUSE`, `NB_CAR`
- **Mobility assets**: `TWO_WHEELER`, `BIKE`, `ELECT_SCOOTER`
- **Transit access**: `SUB` (Navigo), `IMAGINER_SUB`, `OTHER_SUB_PT`, `BIKE_SUB`, `NSM_SUB`

The pipeline applies **vulnerability-aligned feature engineering** (ensuring
higher values consistently indicate higher vulnerability), **nonlinear
transformations** (inverse-log and log(1+x) to model diminishing returns of
resources), and **Principal Component Analysis (PCA)** for data-driven,
objective weighting of components.

### Stage 2 — Agent-Based Model (ABM)

The SVI directly parameterizes each agent's behavioral rules:

- **Activation delay**: higher SVI → longer reaction time before beginning evacuation
- **Speed multiplier**: higher SVI → reduced effective travel speed
- **Patience threshold**: higher SVI → lower tolerance for congestion before failing to reroute

Agents navigate a multi-modal network (OpenStreetMap road/walk/bike graphs +
IDFM GTFS public transit timetables) across a **50 km evacuation radius**
centered on Paris over a **3-hour simulation horizon**. The simulation was run
across **48 parameterized configurations**.

For full methodological details, see [`simulation/README.md`](simulation/README.md).

---

## Publication

**NetMob25 Book of Abstracts:**
> Y. M. Abd ElHameid and N. Gamal ElDin, "A Simulation Study on Equitable
> Mobility During City Emergencies, Focusing on Vulnerable Groups,"
> *NetMob25 Data Challenge*, Paris, 2025.
> [Book of Abstracts — NetMob25](https://netmob.org/www25/files/NetMob25_Book_of_Abstracts.pdf)

<!--**Supplementary outputs (interactive maps, extended figures):**
> Zenodo record: [https://zenodo.org/records/17074045](https://zenodo.org/records/17074045)
-->
**ArXiv:** [ArXiv version coming soon]

---

## Repository Structure

```
Sim-Equity/
│
├── README.md                     This file
├── pyproject.toml                Project and dependency configuration
├── uv.lock                       Locked dependency versions (reproducibility)
│
├── simulation/                   Stage 2: ABM simulation package
│   ├── README.md                 Simulation quick-start and module guide
│   ├── __init__.py
│   ├── preparing_resources.py    Network loading and caching
│   ├── configs/                  Simulation scenario configurations
│   │   ├── evacuation_simulation.json
│   │   └── simulation_summary.json
│   ├── model/                    ABM core: agent class, initializer, analytics
│   │   ├── evacuation_model.py           Agent step logic, SVI-driven behavior
│   │   ├── agents_model_initializer.py   Population instantiation
│   │   ├── simulation_analytics.py       Metrics collection and export
│   │   └── setup.py
│   └── space/                    Spatial environment
│       ├── evacuation_area_initializer.py
│       ├── pre_process_amenities.py
│       └── cache/                Pre-computed route geometries (21 files)
│
├── data/
│   └── maps/                     Geospatial data (OSM, GTFS, road networks)
│       ├── IDFM-gtfs/            IDF public transit timetables (GTFS format)
│       ├── osmnx_layers/         Pre-built walk/bike/drive GraphML networks
│       └── osm_chunks_pyrosm/    Raw OSM extracts for Île-de-France
│
├── analysis/                     Post-simulation analysis code
│   ├── notebooks/
│   │   ├── evacuation_results_analysis.ipynb Equity gap analysis
│   │   ├── public_transport_network.ipynb    GTFS network investigation
│   │   └── social_vulnerability_analysis.ipynb SVI construction walkthrough
│   └── scripts/                  Python script equivalents of notebooks
│       ├── evacuation_results_analysis.py
│       └── public_transport_network.py
│
├── outputs/                      All generated research artifacts
│   ├── simulation_runs/          Per-run JSON metadata (48 runs)
│   ├── agent_states/             Agent-level CSV outputs and journey logs
│   └── figures/                  All generated plots, maps, and visualizations
│       ├── svi_analysis/              SVI distribution and statistical analysis
│       ├── evacuation_analytics/      Equity gap figures, mode/vulnerability plots
│       ├── evacuation_maps/           Interactive HTML + static evacuation maps
│       ├── behavioral_modeling/       SVI → behavioral parameter mappings
│       ├── dimensionality_reduction/  PCA / t-SNE validation plots
│       ├── relationships_with_svi/    Feature × SVI correlation plots
│       └── relationships_with_transformations/  Nonlinear transform visualizations
│
├── scripts/
│   └── main.py                   Simulation entry point
│
└── archive/                      Preserved exploratory and intermediate files
    ├── scratch/                  Temporary scripts (t.py, t2.py)
    ├── root_level_figures/       Earlier-stage duplicate figures
    └── profiling_reports/        ydata-profiling HTML reports
```

---

## Reproducibility

This repository is designed for full research reproducibility.

The simulation was executed across **48 parameterized runs**. Per-run metadata
is stored in `outputs/simulation_runs/evacuation_simulation_{1..48}/`, with
`info.json` and `parameters_constants.json` documenting the exact configuration
for each run.

The aggregate results used in the paper are in
`outputs/agent_states/simulation_outcomes/`. All analysis notebooks in
`analysis/` reproduce the paper's figures from these CSVs and can be re-run
independently.

Interactive choropleth and evacuation trace maps are available locally at
`outputs/figures/evacuation_maps/` and archived on
[Zenodo (record 17074045)](https://zenodo.org/records/17074045) for easy
browsing without local setup.

> **Note on raw data:** The NetMob25 dataset is not redistributed here in
> accordance with its data-use agreement. Researchers may request access at
> [https://netmob.org](https://netmob.org/www25/#data_challenge). The SVI scores derived from the
> dataset are embedded in the simulation configuration and output files.
> The IDFM GTFS timetables are sourced from
> [IDFM Open Data](https://prim.iledefrance-mobilites.fr/en/jeux-de-donnees/offre-horaires-tc-gtfs-idfm).

---

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for fast, reproducible
dependency management.

```bash
# 1. Clone the repository
git clone https://github.com/<your-org>/Sim-Equity.git
cd Sim-Equity

# 2. Install with uv (recommended — uses uv.lock for exact versions)
uv sync

# 3. Alternatively, install with pip
pip install -e .
```

### Core dependencies

| Package                  | Role                                              |
|--------------------------|---------------------------------------------------|
| `mesa`                   | Agent-Based Modeling framework                    |
| `osmnx`                  | OSM street network download and analysis          |
| `networkx`               | Graph algorithms (Dijkstra shortest-path routing) |
| `geopandas` / `pyrosm`   | Geospatial data I/O and processing                |
| `scikit-learn`           | PCA for SVI construction                          |
| `pandas` / `numpy`       | Tabular data processing                           |
| `matplotlib` / `seaborn` | Static visualization                              |
| `folium`                 | Interactive choropleth and trace maps             |
| `pyproj`                 | Coordinate reference system transformations       |

---

## Usage

### Step 1 — Prepare geospatial resources

```bash
python -c "from simulation.preparing_resources import prepare_all; prepare_all()"
```

This validates and caches the OSM network layers (walk, bike, drive) and GTFS
transit data. **Skip this step** if `data/maps/osmnx_layers/` already contains
the pre-built `.graphml` files (included in the repository).

### Step 2 — Construct the SVI

Open and run `analysis/notebooks/social_vulnerability_analysis.ipynb` in Jupyter:

```bash
jupyter notebook analysis/notebooks/social_vulnerability_analysis.ipynb
```

This notebook loads the NetMob25 individual-level data, applies feature
engineering and nonlinear transforms, runs PCA, and writes per-agent SVI scores.
It also produces the SVI distribution figures in `outputs/figures/svi_analysis/`.

### Step 3 — Run the simulation

```bash
python scripts/main.py --config simulation/configs/evacuation_simulation.json
```

Results are written to `outputs/simulation_runs/` and `outputs/agent_states/`.
Estimated runtime: 10–60 minutes per run depending on hardware (48 total runs).

### Step 4 — Analyze results

```bash
# Script
python analysis/scripts/evacuation_results_analysis.py

# Or interactively
jupyter notebook analysis/notebooks/evacuation_results_analysis.ipynb
```

---

## Results

All outputs are **pre-generated and included** in this repository. You do not
need to re-run the simulation to inspect results.

### Key output files

| Output                 | Path                                                                   |
|------------------------|------------------------------------------------------------------------|
| Agent final states     | `outputs/agent_states/final_agent_states.csv`                          |
| Journey segment detail | `outputs/agent_states/simulation_outcomes/Journey_Segments_Detail.csv` |
| Enhanced agent summary | `outputs/agent_states/simulation_outcomes/Enhanced_Agent_Summary.csv`  |
| Per-run metadata (×48) | `outputs/simulation_runs/evacuation_simulation_{1..48}/`               |

### Key paper figures

| Figure                             | Path                                                                            |
|------------------------------------|---------------------------------------------------------------------------------|
| SVI distribution and choropleth    | `outputs/figures/svi_analysis/`                                                 |
| Equity gap and success rates       | `outputs/figures/evacuation_analytics/success_rate_by_vulnerability.png`        |
| Mode × vulnerability heatmap       | `outputs/figures/evacuation_analytics/transportation_mode_by_vulnerability.png` |
| Evacuation time distributions      | `outputs/figures/evacuation_analytics/evacuation_time_by_vulnerability.png`     |
| Interactive agent traces (driving) | `outputs/figures/evacuation_maps/evacuation_map_vehicle.html`                   |
| Interactive agent traces (walking) | `outputs/figures/evacuation_maps/evacuation_map_walking.html`                   |
| Interactive agent traces (cycling) | `outputs/figures/evacuation_maps/evacuation_map_bike.html`                      |

---

## Citation

If you use this code, methodology, or results, please cite:

```bibtex
@inproceedings{abdelhameid2025simequity,
  title     = {A Simulation Study on Equitable Mobility During City Emergencies,
               Focusing on Vulnerable Groups},
  author    = {Abd ElHameid, Youssef M. and Gamal ElDin, Noha},
  booktitle = {NetMob25 Data Challenge},
  year      = {2025},
  address   = {Paris, France},
  url       = {https://zenodo.org/records/17074045}
}
```

---

## License

This repository is released under the **MIT License**. See [`LICENSE`](LICENSE) for full terms.

The **NetMob25 dataset** is subject to its own data-use agreement and is not
redistributed here. The **IDFM GTFS timetables** are sourced from the IDFM Open
Data platform under their open license. OpenStreetMap data © OpenStreetMap
contributors (ODbL).

---

## Contact

|                          |                                    |
|--------------------------|------------------------------------|
| Youssef M. Abd El Hameid | s-youssef.hameid@zewailcity.edu.eg |
| Noha Gamal El Din        | ngamal@nu.edu.eg                   |
