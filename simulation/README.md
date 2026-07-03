# simulation/ — ABM Module Guide

This directory contains the Agent-Based Model (ABM) that implements Stage 2 of
the Sim-Equity methodology: simulating large-scale emergency evacuation in
Île-de-France with Social Vulnerability Index (SVI)-driven agent behavior.

For the full project overview, see the [root README](../README.md).

---

## Quick Start

```bash
# From the project root
python scripts/main.py
```

Expected runtime: 10–60 minutes per run depending on hardware and agent count.
Progress is logged per simulation time step.

Results are written to:

- `outputs/simulation_runs/evacuation_simulation_{run_id}/` — per-run JSON metadata
- `outputs/agent_states/final_agent_states.csv` — agent states at simulation end
- `outputs/agent_states/fallback_agent_states.csv` — agents requiring fallback routing
- `outputs/figures/` — all generated plots

---

## Module Interaction Map

The simulation has a clear dependency chain from data loading through to metrics
export:

```
scripts/main.py                          ← entry point
    │
    ├─ reads: simulation/configs/evacuation_simulation.json
    │         (scenario parameters, SVI behavioral mappings, output paths)
    │
    ├─ calls: simulation/preparing_resources.py
    │         └─ loads OSM GraphML networks into memory
    │            (data/maps/osmnx_layers/*.graphml)
    │         └─ validates IDFM GTFS timetables
    │            (data/maps/IDFM-gtfs/*.csv)
    │
    ├─ calls: simulation/space/pre_process_amenities.py
    │         └─ builds an index of shelters, schools, hospitals
    │            from OSM amenity data
    │            (data/maps/osmnx_layers/idf_amenities.csv)
    │
    ├─ calls: simulation/space/evacuation_area_initializer.py
    │         └─ defines the 50 km evacuation zone centred on Paris
    │         └─ assigns each agent a target safe destination
    │         └─ uses pre-computed geometry cache
    │            (simulation/space/cache/*.json)
    │
    └─ calls: simulation/model/agents_model_initializer.py
              └─ reads SVI scores and mode assignments from config
              └─ instantiates one EvacuationAgent per individual
                 (simulation/model/evacuation_model.py)
                 │
                 Each agent per time step:
                 ├─ checks activation delay (SVI-driven)
                 ├─ selects mode (walk / bike / MPV / PT)
                 ├─ routes via Dijkstra on the mode-appropriate GraphML
                 ├─ updates position, distance, congestion state
                 └─ transitions to ARRIVED / EVACUATING / FAILED
              │
              └─ calls at run end: simulation/model/simulation_analytics.py
                                   └─ computes equity metrics per SVI quartile
                                   └─ writes CSVs to outputs/agent_states/
                                   └─ writes figures to outputs/figures/
```

### Execution flow summary

| Step | Module                           | Action                                    |
|------|----------------------------------|-------------------------------------------|
| 1    | `preparing_resources.py`         | Load and validate all network data        |
| 2    | `pre_process_amenities.py`       | Build shelter/destination index           |
| 3    | `evacuation_area_initializer.py` | Define zone, assign destinations          |
| 4    | `agents_model_initializer.py`    | Instantiate 3,300+ agents with SVI params |
| 5    | `evacuation_model.py` (×N steps) | Simulate each agent per time step         |
| 6    | `simulation_analytics.py`        | Export results and generate figures       |

---

## Configuration Flow

The simulation is controlled by `simulation/configs/evacuation_simulation.json`.

### Key configuration blocks

```jsonc
{
  "scenario": {
    // Geographic parameters
    "crisis_center_lat": 48.8566,      // Paris (Notre-Dame)
    "crisis_center_lon": 2.3522,
    "evacuation_radius_km": 50,        // The evacuation zone is 50 km around the center
    "simulation_horizon_min": 180,     // 3-hour window
    "time_step_sec": 60                // 1-minute resolution
  },

  "agents": {
    "n_agents": 3300,
    // How agent mode choices are initialized:
    // "netmob25_observed" uses observed travel modes from the dataset
    "mode_initialization": "netmob25_observed",
    // SVI weighting scheme used for behavioral mapping:
    "svi_parameterization": "pca_weighted"
  },

  "svi_behavioral_mapping": {
    // Maps each SVI quartile to behavioral parameters.
    // Higher SVI = longer delay, slower speed, less patience.
    "Low":       { "delay_sec": 120, "speed_mult": 1.0,  "patience": 0.9 },
    "Moderate":  { "delay_sec": 300, "speed_mult": 0.85, "patience": 0.7 },
    "High":      { "delay_sec": 600, "speed_mult": 0.70, "patience": 0.5 },
    "Very_High": { "delay_sec": 900, "speed_mult": 0.55, "patience": 0.3 }
  },

  "output": {
    "runs_dir":    "outputs/simulation_runs",
    "states_dir":  "outputs/agent_states",
    "figures_dir": "outputs/figures"
  }
}
```

The behavioral parameters are the core of the SVI-to-simulation linkage:

| Parameter    | Description                                                                      | SVI effect                 |
|--------------|----------------------------------------------------------------------------------|----------------------------|
| `delay_sec`  | Seconds before an agent begins evacuating after the crisis event                 | Higher SVI → longer delay  |
| `speed_mult` | Fraction of the nominal network speed the agent achieves                         | Higher SVI → slower        |
| `patience`   | Fraction of nominal patience before the agent fails to reroute around congestion | Higher SVI → less tolerant |

---

## Input / Output Mapping

### Inputs

| Input           | Path                                                        | Format   | Description                                                   |
|-----------------|-------------------------------------------------------------|----------|---------------------------------------------------------------|
| Walk network    | `data/maps/osmnx_layers/IDF_walk_network.graphml`           | GraphML  | Pedestrian road graph for IDF                                 |
| Bike network    | `data/maps/osmnx_layers/IDF_bike_network.graphml`           | GraphML  | Cycling road graph                                            |
| Drive network   | `data/maps/osmnx_layers/IDF_drive_network.graphml`          | GraphML  | Motorized vehicle road graph                                  |
| Transit network | `data/maps/osmnx_layers/IDF_transportation_network.graphml` | GraphML  | Multimodal combined graph                                     |
| GTFS timetables | `data/maps/IDFM-gtfs/*.csv`                                 | GTFS CSV | IDF public transit schedules (IDFM)                           |
| Amenities       | `data/maps/osmnx_layers/idf_amenities.csv`                  | CSV      | OSM amenity locations (shelters, schools, hospitals)          |
| OSM raw         | `data/maps/osm_chunks_pyrosm/*.osm.pbf`                     | PBF      | Raw OSM extract for IDF (5 chunks)                            |
| Config          | `simulation/configs/evacuation_simulation.json`             | JSON     | Scenario + behavioral parameters                              |
| Summary config  | `simulation/configs/simulation_summary.json`                | JSON     | Aggregate run summary                                         |
| Route cache     | `simulation/space/cache/*.json`                             | JSON     | Pre-computed geometries (21 files, speeds up spatial lookups) |

### Intermediate artifacts

| Artifact             | Location                  | Description                                             |
|----------------------|---------------------------|---------------------------------------------------------|
| Route geometry cache | `simulation/space/cache/` | Cached shortest-path geometries; populated on first run |
| Network objects      | In-memory (not persisted) | OSM GraphML loaded into NetworkX graph objects          |
| Agent population     | In-memory (not persisted) | Mesa Agent objects, one per individual                  |

### Outputs

| Output             | Path                                                                          | Format | Description                                          |
|--------------------|-------------------------------------------------------------------------------|--------|------------------------------------------------------|
| Per-run info       | `outputs/simulation_runs/evacuation_simulation_{i}/info.json`                 | JSON   | Run metadata (timestamp, seed, agent count)          |
| Per-run params     | `outputs/simulation_runs/evacuation_simulation_{i}/parameters_constants.json` | JSON   | Exact parameters for that run                        |
| Final agent states | `outputs/agent_states/final_agent_states.csv`                                 | CSV    | Agent state at end of simulation                     |
| Fallback states    | `outputs/agent_states/fallback_agent_states.csv`                              | CSV    | Agents that triggered fallback routing logic         |
| Trial statistics   | `outputs/agent_states/simulation_outcomes/Agents_Statistics_Trial.csv`        | CSV    | Aggregate per-trial statistics                       |
| Enhanced summary   | `outputs/agent_states/simulation_outcomes/Enhanced_Agent_Summary.csv`         | CSV    | Full per-agent summary with derived metrics          |
| Journey segments   | `outputs/agent_states/simulation_outcomes/Journey_Segments_Detail.csv`        | CSV    | Per-segment journey breakdown (mode, distance, time) |
| Figures            | `outputs/figures/**/*.png`                                                    | PNG    | All analysis plots (organized by sub-category)       |
| Interactive maps   | `outputs/figures/evacuation_maps/*.html`                                      | HTML   | Folium maps with agent traces and SVI coloring       |

### Agent state schema (`final_agent_states.csv`)

| Column                    | Type        | Description                                     |
|---------------------------|-------------|-------------------------------------------------|
| `agent_id`                | str         | Unique agent identifier                         |
| `svi_score`               | float [0,1] | Continuous SVI (higher = more vulnerable)       |
| `svi_quartile`            | str         | Low / Moderate / High / Very_High               |
| `primary_mode`            | str         | Walking / Bike / MPV / PT                       |
| `final_status`            | str         | ARRIVED / EVACUATING / FAILED                   |
| `evacuation_time_min`     | float / NaN | Minutes to reach safe zone (NaN if not arrived) |
| `distance_travelled_km`   | float       | Total network distance covered                  |
| `destination_distance_km` | float       | Great-circle distance to assigned destination   |
| `age`                     | int         | Agent age                                       |
| `sex`                     | int         | Encoded sex (0 = Man, 1 = Woman)                |
| `nb_car`                  | float       | Number of cars in household (transformed)       |
| `pmr`                     | int         | Reduced mobility indicator                      |

Agent terminal states are defined as:

- **ARRIVED**: agent reached a designated safe destination within the simulation horizon
- **EVACUATING**: agent was actively traversing the network at simulation end (right-censored)
- **FAILED**: agent could not find a feasible path or reroute to safety

---

## Results Location

| Result type                    | Directory                                             |
|--------------------------------|-------------------------------------------------------|
| SVI distribution, CDF, Q-Q     | `outputs/figures/svi_analysis/`                       |
| Feature × SVI correlations     | `outputs/figures/relationships_with_svi/`             |
| Nonlinear transform plots      | `outputs/figures/relationships_with_transformations/` |
| PCA / t-SNE validation         | `outputs/figures/dimensionality_reduction/`           |
| SVI → behavioral param plots   | `outputs/figures/behavioral_modeling/`                |
| Evacuation equity analytics    | `outputs/figures/evacuation_analytics/`               |
| Agent trace maps (interactive) | `outputs/figures/evacuation_maps/`                    |
| Aggregate run logs             | `outputs/simulation_runs/`                            |
| Agent-level CSV data           | `outputs/agent_states/`                               |

---

## Extending Experiments

### Adding a new scenario configuration

```bash
# 1. Copy the base config
cp simulation/configs/evacuation_simulation.json \
   simulation/configs/evacuation_simulation_scenario2.json

# 2. Edit the new config (e.g., change radius, time horizon, or SVI mappings)
nano simulation/configs/evacuation_simulation_scenario2.json

# 3. Run
python scripts/main.py --config simulation/configs/evacuation_simulation_scenario2.json
```

### Modifying SVI behavioral parameters

Edit the `svi_behavioral_mapping` block in the config JSON. The four quartiles
(`Low`, `Moderate`, `High`, `Very_High`) each have three behavioral parameters
(`delay_sec`, `speed_mult`, `patience`). These are the primary levers for
sensitivity analysis.

To use a continuous (non-quartile) SVI mapping, modify
`simulation/model/agents_model_initializer.py` — specifically the function that
translates `svi_score` into agent attributes.

### Modifying agent step logic

The per-timestep agent behavior is in `simulation/model/evacuation_model.py`,
in the `step()` method. The logic sequence is:

```
step():
  1. Check if activation_delay has elapsed → if not, remain idle
  2. If at destination → mark ARRIVED, stop
  3. Compute next move on mode-appropriate graph
  4. Check patience → if congestion exceeds threshold, mark FAILED or reroute
  5. Advance position, update distance, update time
```

### Adding a new transportation mode

1. Prepare the mode's OSM network and save it to `data/maps/osmnx_layers/` as a `.graphml` file.
2. Register it in `simulation/preparing_resources.py` by adding it to the `NETWORK_MODES` dictionary.
3. Add the mode's behavioral profile to the config JSON.
4. Handle the new mode in the agent `step()` logic (routing and speed lookup).

### Adding new equity metrics

Extend `simulation/model/simulation_analytics.py`. The analytics module receives
the full agent population as a list of `EvacuationAgent` objects at run end and
can compute any derived metric from their attributes. Add a new function and call
it from the `run_analytics()` entry point.

---

## Module Reference

| Module                                 | Key responsibility                                            |
|----------------------------------------|---------------------------------------------------------------|
| `preparing_resources.py`               | Load OSM GraphML and GTFS; validate data completeness         |
| `space/evacuation_area_initializer.py` | Define evacuation zone polygon; assign agent destinations     |
| `space/pre_process_amenities.py`       | Build geospatial index of shelters and safe destinations      |
| `model/agents_model_initializer.py`    | Read config + SVI scores; instantiate Mesa Agent objects      |
| `model/evacuation_model.py`            | Agent class: `__init__`, `step()`, routing, state transitions |
| `model/simulation_analytics.py`        | Post-simulation metrics, figure generation, CSV export        |
| `model/setup.py`                       | Package setup and optional compilation hooks                  |
