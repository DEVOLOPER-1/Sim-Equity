# FILE: main.py
import gc
from datetime import datetime

import polars as pl
from tqdm import tqdm

from simulation.model.agents_model_initializer import AgentsGatherer
from simulation.model.evacuation_model import EvacuationModel
from simulation.model.simulation_analytics import SimulationAnalytics
from simulation.space.evacuation_area_initializer import EnvironmentInitializer


def main():
    # Configuration
    SCENARIO_CENTER_LAT = 48.858844
    SCENARIO_CENTER_LON = 2.347012
    SCENARIO_RADIUS_KM = 50.0
    SCENARIO_START_DATETIME = datetime(2023, 1, 10, 16, 0, 0)
    MAX_SIMULATION_STEPS = 60
    STEP_SECONDS = 60
    SVI_SPEED_PENALTY = 0.5
    MAX_SVI_START_DELAY_S = 1800
    BASE_PATIENCE_S = 300
    DATA_DIR = "simulation/maps_data/osmnx_layers/"

    print(f"--- SIMULATION STARTING --- {datetime.now()}")

    # Initialize environment
    env = EnvironmentInitializer(
        (SCENARIO_CENTER_LAT, SCENARIO_CENTER_LON), SCENARIO_RADIUS_KM
    )
    evacuation_area_polygon = env.get_made_polygon
    gc.collect()

    # Gather agents
    agents_gatherer = AgentsGatherer(
        evacuation_area_polygon=evacuation_area_polygon,
        evac_area_center=(SCENARIO_CENTER_LAT, SCENARIO_CENTER_LON),
        time="10:18:21:00",
    )
    gc.collect()

    agents_gatherer.read_and_summarize_agents(
        fallback_to_full_trace=False, verbose=False
    )
    agents_df = pl.read_csv("data/mesa_initializers.csv")

    if agents_df.is_empty():
        print(f"!!! No agents in evacuation zone !!! {datetime.now()}")
        return

    print(f"-> Found {agents_df.shape[0]} agents {datetime.now()}")

    # Load graphs
    # print(f"Loading network graphs... {datetime.now()}")
    # G_drive = rx.read_graphml()[0]
    # G_walk = rx.read_graphml()[0]
    # G_cycle = rx.read_graphml()[0]

    # Load amenities
    amenities_df = pl.read_csv(DATA_DIR + "idf_amenities.csv")
    gc.collect()

    print(f"-> Data assets loaded {datetime.now()}")

    # Create model
    print(f"Instantiating model... {datetime.now()}")
    model = EvacuationModel(
        agents_df=agents_df,
        graphml_path_drive=(DATA_DIR + "IDF_drive_network.graphml"),
        graphml_path_walk=DATA_DIR + "IDF_walk_network.graphml",
        graphml_path_cycle=DATA_DIR + "IDF_bike_network.graphml",
        amenities_df=amenities_df,
        evacuation_area_polygon=evacuation_area_polygon,
        start_datetime=SCENARIO_START_DATETIME,
        step_seconds=STEP_SECONDS,
        svi_speed_penalty=SVI_SPEED_PENALTY,
        max_svi_start_delay_s=MAX_SVI_START_DELAY_S,
        base_patience_s=BASE_PATIENCE_S,
    )
    print(f"-> Model instantiated {datetime.now()}")

    # Run simulation
    print(f"Running simulation for {MAX_SIMULATION_STEPS} steps... {datetime.now()}")
    for i in tqdm(range(MAX_SIMULATION_STEPS)):
        model.step()
    print("-> Simulation complete")

    # Process results
    print(f"Extracting results... {datetime.now()}")
    model_df = pl.DataFrame(model.datacollector.get_model_vars_dataframe())
    raw_agent_df = model.datacollector.get_agent_vars_dataframe().reset_index()
    final_agent_state_df = pl.DataFrame(
        raw_agent_df.groupby("AgentID").last().reset_index()
    )

    # Generate analytics
    print(f"Generating analytics... {datetime.now()}")
    analytics = SimulationAnalytics(
        model_data=model_df, agent_data=final_agent_state_df
    )

    analytics.analyze_svi_vs_outcome()
    analytics.plot_svi_vs_evacuation_time(save=True)
    analytics.plot_equity_gap(save=True)
    analytics.plot_bottleneck_map(model.hybrid_managers["CAR"], save=True)

    print(f"--- ANALYSIS COMPLETE --- {datetime.now()}")


if __name__ == "__main__":
    main()
