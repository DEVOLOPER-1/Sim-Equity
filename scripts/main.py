# FILE: main.py
import gc
import json
import os
import time
from datetime import datetime
from pathlib import Path

# Set matplotlib backend early
import matplotlib
import polars as pl

matplotlib.use("Agg")  # Force Agg backend for saving without display

from simulation.model.agents_model_initializer import AgentsGatherer
from simulation.model.evacuation_model import run_simulation
from simulation.model.simulation_analytics import SimulationAnalytics
from simulation.space.evacuation_area_initializer import EnvironmentInitializer


class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super(DateTimeEncoder, self).default(obj)


def serialize_results_dict(results_dict):
    """Recursively convert objects to JSON-serializable forms."""
    import polars as pl
    import pandas as pd
    from shapely.geometry.base import BaseGeometry
    
    if isinstance(results_dict, dict):
        return {
            key: serialize_results_dict(value) for key, value in results_dict.items()
        }
    elif isinstance(results_dict, list):
        return [serialize_results_dict(item) for item in results_dict]
    elif isinstance(results_dict, datetime):
        return results_dict.isoformat()
    elif isinstance(results_dict, (pl.DataFrame, pd.DataFrame)):
        return results_dict.to_dicts() if hasattr(results_dict, "to_dicts") else results_dict.to_dict(orient="records")
    elif isinstance(results_dict, BaseGeometry):
        return results_dict.wkt
    elif isinstance(results_dict, (str, int, float, bool, type(None))):
        return results_dict
    else:
        try:
            # Test if it is natively serializable
            json.dumps(results_dict)
            return results_dict
        except TypeError:
            return str(results_dict)


def save_results_manually(results, filename="evacuation_simulation"):
    """Save AgentPy results manually with proper serialization to the simulation/configs directory."""
    try:
        # Convert results to dictionary
        if hasattr(results, "_data"):
            results_dict = results._data
        elif hasattr(results, "to_dict"):
            results_dict = results.to_dict()
        else:
            results_dict = dict(results)

        # Serialize datetime objects and other complex fields
        serialized_dict = serialize_results_dict(results_dict)

        # Save to simulation/configs directory
        output_dir = Path(__file__).parent.parent / "simulation" / "configs"
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / f"{filename}.json"

        with open(filepath, "w") as f:
            json.dump(serialized_dict, f, indent=2, cls=DateTimeEncoder)

        print(f"Results saved to {filepath}")
        return True

    except Exception as e:
        print(f"Error saving results: {e}")
        return False



def main():
    # Configuration
    SCENARIO_CENTER_LAT = 48.858844
    SCENARIO_CENTER_LON = 2.347012
    SCENARIO_RADIUS_KM = 50.0
    MAX_SIMULATION_STEPS = 180
    DATA_DIR = str(Path(__file__).parent.parent / "data" / "maps" / "osmnx_layers") + "/"

    print(f"--- SIMULATION STARTING --- {datetime.now()}")

    # Initialize environment - center is passed as (lat, lon)
    env = EnvironmentInitializer(
        (SCENARIO_CENTER_LAT, SCENARIO_CENTER_LON), SCENARIO_RADIUS_KM
    )
    evacuation_area_polygon = env.get_made_polygon
    gc.collect()

    # Gather agents - center is passed as (lat, lon)
    agents_gatherer = AgentsGatherer(
        evacuation_area_polygon=evacuation_area_polygon,
        evac_area_center=(SCENARIO_CENTER_LAT, SCENARIO_CENTER_LON),
        time="10:18:21:00",
    )
    gc.collect()

    agents_gatherer.read_and_summarize_agents(fallback_to_full_trace=True, verbose=True)
    agents_df = pl.read_csv("data/mesa_initializers.csv")

    if agents_df.is_empty():
        print(f"!!! No agents in evacuation zone !!! {datetime.now()}")
        return

    print(f"-> Found {agents_df.shape[0]} agents {datetime.now()}")

    # Load amenities
    amenities_df = pl.read_csv(DATA_DIR + "idf_amenities.csv")
    gc.collect()

    print(f"-> Data assets loaded {datetime.now()}")

    # Define parameters
    parameters = {
        "start_datetime": datetime(2023, 1, 1, 8, 0, 0),  # Simulation start time
        "step_seconds": 60,  # 1 minute per step
        "svi_speed_penalty": 0.5,
        "max_svi_start_delay_s": 1800,  # 30 minutes max delay
        "base_patience_s": 300,  # 5 minutes base patience
        "graphml_path_drive": DATA_DIR + "IDF_drive_network.graphml",
        "graphml_path_walk": DATA_DIR + "IDF_walk_network.graphml",
        "graphml_path_cycle": DATA_DIR + "IDF_bike_network.graphml",
        "amenities_df": amenities_df,
        "evacuation_area_polygon": evacuation_area_polygon,
        "agents_df": agents_df,
        "steps": MAX_SIMULATION_STEPS,  # 60 steps = 1 hour
        "use_public_transport": True,
        "osm_pbf_path": DATA_DIR + "ile-de-france-250902.osm.pbf",
        "gtfs_zip_path": DATA_DIR + "IDFM-gtfs.zip",
    }

    # Access and analyze results
    print(f"Running simulation for {MAX_SIMULATION_STEPS} steps... {datetime.now()}")
    start_time = time.time()

    _SIM_OUTCOMES = str(Path(__file__).parent.parent / "outputs" / "agent_states" / "simulation_outcomes")
    os.makedirs(_SIM_OUTCOMES, exist_ok=True)
    os.makedirs(_SIM_OUTCOMES + "/agents_traces", exist_ok=True)
    # Run the simulation
    model, results = run_simulation(parameters)

    end_time = time.time()
    print(f"-> Simulation completed in {end_time - start_time:.2f} seconds")

    # Access and analyze results
    print("\n=== SIMULATION RESULTS ===")

    # Count agent statuses
    from collections import Counter

    status_counts = Counter(agent.status for agent in model.agents)

    # Also count failure reasons if needed
    fail_reasons = Counter(
        agent.fail_reason for agent in model.agents if agent.status == "FAILED"
    )

    for status, count in status_counts.items():
        print(f"{status}: {count} agents")

    if fail_reasons:
        print("\nFailure reasons:")
        for reason, count in fail_reasons.items():
            print(f"  {reason}: {count} agents")

    # Save results with proper datetime handling
    print(f"Saving results... {datetime.now()}")

    # Try AgentPy's built-in save first (it might work if results don't contain datetime)
    try:
        results.save(exp_name="evacuation_simulation")
        print("-> Results saved using AgentPy's built-in method")
    except TypeError as e:
        if "datetime" in str(e):
            print(
                "-> AgentPy save failed due to datetime objects, using manual save..."
            )
            save_results_manually(results, "evacuation_simulation")
        else:
            print(f"-> AgentPy save failed with error: {e}")
            save_results_manually(results, "evacuation_simulation")

    # Also save a summary of key metrics
    summary_data = {
        "simulation_metadata": {
            "start_time": parameters["start_datetime"].isoformat(),
            "total_steps": MAX_SIMULATION_STEPS,
            "step_duration_seconds": parameters["step_seconds"],
            "total_agents": len(model.agents),
            "completion_time_seconds": end_time - start_time,
        },
        "agent_status_summary": dict(status_counts),
        "failure_reasons": dict(fail_reasons) if fail_reasons else {},
    }

    summary_path = Path(__file__).parent.parent / "simulation" / "configs" / "simulation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2, cls=DateTimeEncoder)

    print(f"-> Summary saved to {summary_path}")

    # Generate analytics with the updated module
    print(f"Generating analytics... {datetime.now()}")

    try:
        # Create analytics instance directly from the model
        analytics = SimulationAnalytics(model=model)

        # Generate comprehensive summary report
        analytics.generate_summary_report()

        # Generate all visualizations
        print("Creating visualizations...")

        # 1. Agent status distribution
        analytics.plot_agent_status_distribution(
            save=True, save_kwargs={"filename": "agent_status_distribution.png"}
        )

        # 2. Transportation mode analysis
        analytics.plot_transportation_mode_analysis(
            save=True, save_kwargs={"filename": "transportation_mode_analysis.png"}
        )

        # 3. SVI vs evacuation time (if successful agents exist)
        analytics.plot_svi_vs_evacuation_time(
            save=True, save_kwargs={"filename": "svi_vs_evacuation_time.png"}
        )

        # 4. Equity gap analysis
        analytics.plot_equity_gap(
            save=True, save_kwargs={"filename": "evacuation_equity_gap.png"}
        )

        # 5. Check if bottleneck data exists
        if len(analytics.bottleneck_df) > 0:
            print(f"Found {len(analytics.bottleneck_df)} bottleneck records")
            try:
                print("Creating bottleneck map...")
                analytics.plot_bottleneck_map(
                    save=True, save_kwargs={"filename": "bottleneck_map.png"}
                )
            except Exception as e:
                print(f"-> Bottleneck map generation failed: {e}")
                import traceback

                traceback.print_exc()
        else:
            print("-> No bottleneck data available for mapping")

        print("-> Basic visualizations generated successfully")

        # After existing analytics calls
        print("Creating trace visualizations...")

        # Plot traces for each mode
        for mode in ["CAR", "WALKING", "BIKE"]:
            try:
                print(f"Creating agent traces for {mode}...")
                analytics.plot_agent_traces(
                    mode_filter=mode,
                    save=True,
                    save_kwargs={"filename": f"agent_traces_{mode.lower()}.png"},
                )
            except Exception as e:
                print(f"-> Agent trace visualization for {mode} failed: {e}")
                import traceback

                traceback.print_exc()

        # Plot road usage heatmaps
        for mode in ["CAR", "WALKING", "BIKE"]:
            try:
                print(f"Creating road usage heatmap for {mode}...")
                analytics.plot_road_usage_heatmap(
                    mode=mode,
                    save=True,
                    save_kwargs={"filename": f"road_usage_{mode.lower()}.png"},
                )
            except Exception as e:
                print(f"-> Road usage heatmap for {mode} failed: {e}")
                import traceback

                traceback.print_exc()

        # Check for path_history in agents
        print("Checking agent path history data...")
        agents_with_history = sum(
            1 for a in model.agents if hasattr(a, "path_history") and a.path_history
        )
        print(
            f"-> {agents_with_history} out of {len(model.agents)} agents have path history"
        )
        if agents_with_history > 0:
            sample_agent = next(
                (
                    a
                    for a in model.agents
                    if hasattr(a, "path_history") and a.path_history
                ),
                None,
            )
            if sample_agent:
                print(f"Sample path history: {sample_agent.path_history[:3]}")

        # Save agent states as CSV for further analysis
        analytics.agent_df.write_csv(str(Path(__file__).parent.parent / "outputs" / "agent_states" / "final_agent_states.csv"))
        print("-> Final agent states saved to final_agent_states.csv")

        # Save bottleneck data if available
        if len(analytics.bottleneck_df) > 0:
            analytics.bottleneck_df.write_csv("bottleneck_data.csv")
            print("-> Bottleneck data saved to bottleneck_data.csv")

    except Exception as e:
        print(f"-> Analytics generation failed: {e}")
        import traceback

        traceback.print_exc()

        # Fallback: save basic agent data
        print("-> Creating fallback agent data export...")
        agent_states = []
        for agent in model.agents:
            agent_state = {
                "agent_id": agent.id,
                "status": agent.status,
                "svi": getattr(agent, "svi", 0.0),
                "main_mode": getattr(agent, "main_mode", "UNKNOWN"),
                "evacuation_time": getattr(agent, "evacuation_time", 0),
                "fail_reason": getattr(agent, "fail_reason", None),
            }
            agent_states.append(agent_state)

        agent_states_df = pl.DataFrame(agent_states)
        agent_states_df.write_csv(str(Path(__file__).parent.parent / "outputs" / "agent_states" / "fallback_agent_states.csv"))
        print("-> Fallback agent states saved to fallback_agent_states.csv")

    print(f"--- ANALYSIS COMPLETE --- {datetime.now()}")


if __name__ == "__main__":
    main()
