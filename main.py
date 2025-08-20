# FILE: main.py
import gc
import json
import time
from datetime import datetime

import polars as pl

from simulation.model.agents_model_initializer import AgentsGatherer
from simulation.model.evac_mod_agent_py import run_simulation
from simulation.model.simulation_analytics import SimulationAnalytics
from simulation.space.evacuation_area_initializer import EnvironmentInitializer


class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super(DateTimeEncoder, self).default(obj)


def serialize_results_dict(results_dict):
    """Recursively convert datetime objects to ISO strings in a dictionary."""
    if isinstance(results_dict, dict):
        return {
            key: serialize_results_dict(value) for key, value in results_dict.items()
        }
    elif isinstance(results_dict, list):
        return [serialize_results_dict(item) for item in results_dict]
    elif isinstance(results_dict, datetime):
        return results_dict.isoformat()
    else:
        return results_dict


def save_results_manually(results, filename="evacuation_simulation"):
    """Save AgentPy results manually with datetime serialization."""
    try:
        # Convert results to dictionary
        if hasattr(results, "_data"):
            results_dict = results._data
        elif hasattr(results, "to_dict"):
            results_dict = results.to_dict()
        else:
            results_dict = dict(results)

        # Serialize datetime objects
        serialized_dict = serialize_results_dict(results_dict)

        # Save to JSON file
        with open(f"{filename}.json", "w") as f:
            json.dump(serialized_dict, f, indent=2, cls=DateTimeEncoder)

        print(f"Results saved to {filename}.json")
        return True

    except Exception as e:
        print(f"Error saving results: {e}")
        return False


def main():
    # Configuration
    SCENARIO_CENTER_LAT = 48.858844
    SCENARIO_CENTER_LON = 2.347012
    SCENARIO_RADIUS_KM = 50.0
    MAX_SIMULATION_STEPS = 5
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
    agents_df = pl.read_csv("data/mesa_initializers.csv").limit(
        50
    )  # TODO: REMOVE LIMIT LATER

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
    }

    # Access and analyze results
    print(f"Running simulation for {MAX_SIMULATION_STEPS} steps... {datetime.now()}")
    start_time = time.time()

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

    with open("simulation_summary.json", "w") as f:
        json.dump(summary_data, f, indent=2, cls=DateTimeEncoder)

    print("-> Summary saved to simulation_summary.json")

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

        # 5. Bottleneck map (if bottleneck data exists and graphs are available)
        if len(analytics.bottleneck_df) > 0:
            analytics.plot_bottleneck_map(
                save=True, save_kwargs={"filename": "bottleneck_map.png"}
            )
        else:
            print("-> No bottleneck data available for mapping")

        print("-> All visualizations generated successfully")

        # Save agent states as CSV for further analysis
        analytics.agent_df.write_csv("final_agent_states.csv")
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
        agent_states_df.write_csv("fallback_agent_states.csv")
        print("-> Fallback agent states saved to fallback_agent_states.csv")

    print(f"--- ANALYSIS COMPLETE --- {datetime.now()}")


if __name__ == "__main__":
    main()
