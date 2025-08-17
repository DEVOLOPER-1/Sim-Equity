# FILE: main.py
# -----------------------------
# This is the main execution script for the evacuation simulation.
# It acts as a control panel, handling:
#  1. Configuration of the simulation scenario.
#  2. Initialization of the environment and agents.
#  3. Loading of pre-processed data assets.
#  4. Instantiation and execution of the Mesa model.
#  5. Collection and analysis of the final simulation results.

import gc
from datetime import datetime
from typing import Any

import osmnx as ox
import polars as pl
from tqdm import tqdm  # For a nice progress bar during the simulation run

from simulation.model.agents_model_initializer import AgentsGatherer

# --- Import your custom modules ---
# Ensure these files are in the same directory or a properly configured path
from simulation.model.evacuation_model import EvacuationModel
from simulation.model.simulation_analytics import SimulationAnalytics
from simulation.space.evacuation_area_initializer import EnvironmentInitializer


def select_amenities_out_evacuation_area(
    amenities_df: pl.DataFrame,
    agents_gatherer_object: AgentsGatherer,
    selected: list[Any],
):
    for amenity in amenities_df.iter_rows(named=True):
        if not agents_gatherer_object.are_coords_in_the_evacuation_area(
            (amenity.get("latitude"), amenity.get("longitude"))
        ):
            selected.append(amenity)
    return pl.DataFrame(selected)


def main():
    """Main function to run the entire simulation and analysis pipeline."""

    # --- 1. CONFIGURATION ---
    # All user-adjustable parameters are here. Change these to test different scenarios.

    # -- Scenario Parameters --
    SCENARIO_CENTER_LAT = 48.858844  # Center of Paris (e.g., Châtelet)
    SCENARIO_CENTER_LON = 2.347012
    SCENARIO_RADIUS_KM = 50.0  # 50km radius evacuation zone
    SCENARIO_START_DATETIME = datetime(2023, 1, 10, 16, 0, 0)  # A Tuesday at 4:00 PM

    # -- Simulation Run Parameters --
    MAX_SIMULATION_STEPS = 180  # Number of steps to run the simulation for
    STEP_SECONDS = 60  # Each step represents 60 seconds (1 minute)
    # Total simulation time = 180 steps * 60s/step = 10800s = 3 hours

    # -- Model Behavioral Parameters (The "Tuning Knobs") --
    SVI_SPEED_PENALTY = 0.5  # Max speed reduction for most vulnerable (50%)
    MAX_SVI_START_DELAY_S = 1800  # Max reaction delay for most vulnerable (30 mins)
    BASE_PATIENCE_S = 300  # Patience of least vulnerable before rerouting (5 mins)

    # -- File Paths for Pre-processed Data --
    DATA_DIR = "simulation/maps_data/osmnx_layers/"
    DRIVE_GRAPH_PATH = DATA_DIR + "IDF_drive_network.graphml"
    WALK_GRAPH_PATH = DATA_DIR + "IDF_walk_network.graphml"
    CYCLE_GRAPH_PATH = DATA_DIR + "IDF_bike_network.graphml"
    # Assuming you created this
    AMENITIES_PATH = DATA_DIR + "idf_amenities.csv"

    print("--- SIMULATION STARTING ---")
    print(f"Scenario: Evacuation from a {SCENARIO_RADIUS_KM}km radius around Châtelet.")
    print(f"Start Time: {SCENARIO_START_DATETIME.isoformat()}")

    # --- 2. INITIALIZE ENVIRONMENT AND AGENTS ---
    # This step defines the disaster zone and finds which of the 3300+ people
    # are inside it at the start time.
    print("\nStep 1: Initializing environment and gathering agents...")

    env = EnvironmentInitializer(
        (SCENARIO_CENTER_LAT, SCENARIO_CENTER_LON), SCENARIO_RADIUS_KM
    )
    evacuation_area_polygon = env.get_made_polygon

    gc.collect()

    agents_gatherer = AgentsGatherer(
        evacuation_area_polygon=evacuation_area_polygon,
        evac_area_center=(SCENARIO_CENTER_LAT, SCENARIO_CENTER_LON),
        time="10:18:21:00",
    )

    gc.collect()

    agents_gatherer.read_and_summarize_agents(
        fallback_to_full_trace=False, verbose=False
    )

    gc.collect()

    # This is the crucial DataFrame that will seed our simulation
    agents_df = pl.read_csv("data/mesa_initializers.csv")

    gc.collect()

    if agents_df.is_empty():
        print(
            "\n!!! No agents found in the evacuation zone at the specified time. Halting. !!!"
        )
        return

    print(f"-> Found {agents_df.shape[0]} agents inside the evacuation zone.")

    # --- 3. LOAD PRE-PROCESSED DATA ASSETS ---
    # We load the lightweight, clean graphs and amenity data that you
    # created with your pre-processing scripts.
    print("\nStep 2: Loading pre-processed network graphs and amenities...")

    G_drive = ox.load_graphml(DRIVE_GRAPH_PATH)
    G_walk = ox.load_graphml(WALK_GRAPH_PATH)
    G_cycle = ox.load_graphml(CYCLE_GRAPH_PATH)
    selected = []

    amenities_df = select_amenities_out_evacuation_area(
        pl.read_csv(AMENITIES_PATH), agents_gatherer, selected
    )
    del selected

    print(f"-> Found {amenities_df.shape[0]} shelters outside the evacuation zone.")

    gc.collect()

    print("-> Data assets loaded successfully.")

    # --- 4. INSTANTIATE THE MESA MODEL ---
    # This is where we create the simulation world and populate it with our agents.
    print("\nStep 3: Instantiating the Evacuation Model...")

    model = EvacuationModel(
        agents_df=agents_df,
        G_drive=G_drive,
        G_walk=G_walk,
        G_cycle=G_cycle,
        amenities_df=amenities_df,
        evacuation_area_polygon=evacuation_area_polygon,
        start_datetime=SCENARIO_START_DATETIME,
        step_seconds=STEP_SECONDS,
        svi_speed_penalty=SVI_SPEED_PENALTY,
        max_svi_start_delay_s=MAX_SVI_START_DELAY_S,
        base_patience_s=BASE_PATIENCE_S,
    )

    print("-> Model instantiated successfully.")

    # --- 5. RUN THE SIMULATION ---
    # This is the main simulation loop. We use tqdm for a progress bar.
    print(f"\nStep 4: Running simulation for {MAX_SIMULATION_STEPS} steps...")

    for i in tqdm(range(MAX_SIMULATION_STEPS)):
        model.step()

    print("-> Simulation run complete.")

    # --- 6. EXTRACT DATA FROM THE MODEL ---
    # The Mesa DataCollector has been logging everything. Now we pull it out.
    print("\nStep 5: Extracting simulation results...")

    # The model_df contains our bottleneck log for each step
    model_df = model.datacollector.get_model_vars_dataframe()

    # The agent_df contains the final state of every agent
    # We need to get the *last* recorded state for each agent.
    raw_agent_df = model.datacollector.get_agent_vars_dataframe().reset_index()
    final_agent_state_df = raw_agent_df.groupby("AgentID").last().reset_index()

    print("-> Results extracted successfully.")

    # --- 7. ANALYZE AND VISUALIZE RESULTS ---
    # This is where we use our dedicated analytics class to generate the
    # key insights and plots needed for the report.
    print("\nStep 6: Generating analytics and visualizations...")

    analytics = SimulationAnalytics(
        model_data=model_df, agent_data=final_agent_state_df
    )

    # --- Generate each insight one by one ---

    print("\n--- Insight 1: SVI vs. Evacuation Outcome ---")
    analytics.analyze_svi_vs_outcome()
    analytics.plot_svi_vs_evacuation_time()

    print("\n--- Insight 2: The Evacuation Equity Gap ---")
    analytics.plot_equity_gap()

    print("\n--- Insight 3: Geospatial Bottleneck Analysis ---")
    analytics.plot_bottleneck_map(G_drive)

    print("\n--- ANALYSIS COMPLETE ---")


if __name__ == "__main__":
    main()
