# FILE: simulation_analytics.py
# -----------------------------
import pathlib
from functools import wraps
from typing import Any, Callable, Optional

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import polars as pl
import seaborn as sns


class SimulationAnalytics:
    def __init__(
        self,
        model=None,
        agent_data: Optional[pl.DataFrame] = None,
        model_data: Optional[pl.DataFrame] = None,
    ):
        """
        Initialize the analytics suite with data from a completed AgentPy simulation.

        Args:
            model: The completed AgentPy model instance with agents
            agent_data: DataFrame with agent final states (optional if model provided)
            model_data: DataFrame from model reporters (optional)
        """
        if model is not None:
            # Extract data from AgentPy model
            self.agent_df = self._extract_agent_data_from_model(model)
            self.bottleneck_df = self._extract_bottleneck_data_from_model(model)
            self.model = model
        elif agent_data is not None:
            # Use provided DataFrames
            self.agent_df = agent_data
            self.bottleneck_df = (
                self._process_bottlenecks(model_data)
                if model_data is not None
                else pl.DataFrame()
            )
            self.model = None
        else:
            raise ValueError("Either model or agent_data must be provided")

        print(f"Analytics initialized with {len(self.agent_df)} agent records.")

    def _extract_agent_data_from_model(self, model) -> pl.DataFrame:
        """Extract agent data from AgentPy model into a DataFrame."""
        agent_records = []

        for agent in model.agents:
            record = {
                "agent_id": agent.id,
                "status": agent.status,
                "SVI": getattr(agent, "svi", 0.0),
                "main_mode": getattr(agent, "main_mode", "UNKNOWN"),
                "evacuation_time": getattr(agent, "evacuation_time", 0),
                "fail_reason": getattr(agent, "fail_reason", None),
                "speed_m_s": getattr(agent, "speed_m_s", 0.0),
                "patience_threshold_s": getattr(agent, "patience_threshold_s", 0.0),
                "replan_attempts": getattr(agent, "replan_attempts", 0),
            }
            # Add any additional agent-specific data
            if hasattr(agent, "agent_data"):
                record.update(agent.agent_data)

            agent_records.append(record)

        return pl.DataFrame(agent_records)

    def _extract_bottleneck_data_from_model(self, model) -> pl.DataFrame:
        """Extract bottleneck data from AgentPy model."""
        if hasattr(model, "bottleneck_log") and model.bottleneck_log:
            # Convert bottleneck log to DataFrame
            bottleneck_records = []
            for entry in model.bottleneck_log:
                record = {
                    "time": entry.get("time"),
                    "edge_nodes": str(entry.get("edge_nodes", "")),
                    "load": entry.get("load", 0),
                    "capacity": entry.get("capacity", 1),
                    "congestion_index": entry.get("congestion_index", 0.0),
                    "avg_svi_stuck": entry.get("avg_svi_stuck", 0.0),
                }
                bottleneck_records.append(record)

            return pl.DataFrame(bottleneck_records)
        else:
            # Return empty DataFrame with expected schema
            return pl.DataFrame(
                schema={
                    "time": pl.Datetime,
                    "edge_nodes": pl.String,
                    "load": pl.Int64,
                    "capacity": pl.Int64,
                    "congestion_index": pl.Float64,
                    "avg_svi_stuck": pl.Float64,
                }
            )

    def autosave(func: Callable) -> Callable:
        """
        Decorator for plot methods to add optional saving behaviour.
        """

        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Extract save kwargs
            save = bool(kwargs.pop("save", False))
            save_kwargs = kwargs.pop("save_kwargs", None) or {}

            # Call the original plotting function
            result = func(*args, **kwargs)

            # If saving requested, determine the Figure to save
            if save:
                fig = None

                # Case 1: function returned a Figure
                if isinstance(result, plt.Figure):
                    fig = result
                # Case 2: function returned (fig, ax) or [fig, ax]
                elif (
                    isinstance(result, (tuple, list))
                    and len(result) > 0
                    and isinstance(result[0], plt.Figure)
                ):
                    fig = result[0]
                # Case 3: function returned an Axes-like object (has .figure)
                elif hasattr(result, "figure") and isinstance(
                    result.figure, plt.Figure
                ):
                    fig = result.figure
                # Fallback: current figure
                else:
                    try:
                        fig = plt.gcf()
                    except Exception:
                        fig = None

                # Save the figure
                if fig is not None:
                    if len(args) > 0:
                        self_obj = args[0]
                        if hasattr(self_obj, "save_figure") and callable(
                            getattr(self_obj, "save_figure")
                        ):
                            self_obj.save_figure(fig=fig, **save_kwargs)
                        else:
                            self_obj._save_figure_default(
                                fig, func.__name__, **save_kwargs
                            )
                    else:
                        _save_figure_static(fig, func.__name__, **save_kwargs)

            return result

        return wrapper

    def _save_figure_default(self, fig, func_name: str, **save_kwargs):
        """Default figure saving method."""
        out_dir = save_kwargs.get("folder", "plots")
        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
        fname = save_kwargs.get("filename", f"{func_name}.png")
        dpi = save_kwargs.get("dpi", 300)
        fig.savefig(f"{out_dir}/{fname}", dpi=dpi, bbox_inches="tight")
        print(f"Plot saved to {out_dir}/{fname}")

    def _process_bottlenecks(self, model_data: pl.DataFrame) -> pl.DataFrame:
        """Process raw bottleneck data into a structured DataFrame."""
        if model_data is None or model_data.is_empty():
            return pl.DataFrame(
                schema={
                    "edge_nodes": pl.String,
                    "congestion_index": pl.Float64,
                    "avg_svi_stuck": pl.Float64,
                }
            )

        all_logs = []
        for row in model_data.iter_rows(named=True):
            bottlenecks = row.get("bottlenecks", [])
            if bottlenecks:
                for log_entry in bottlenecks:
                    all_logs.append(log_entry)

        if not all_logs:
            return pl.DataFrame(
                schema={
                    "edge_nodes": pl.String,
                    "congestion_index": pl.Float64,
                    "avg_svi_stuck": pl.Float64,
                }
            )

        return pl.DataFrame(all_logs)

    # --- INSIGHT 1: SVI vs. EVACUATION OUTCOME ---
    def analyze_svi_vs_outcome(self):
        """Calculates and prints key stats about SVI and evacuation success."""
        # Map AgentPy statuses to analysis categories
        arrived_agents = self.agent_df.filter(pl.col("status") == "ARRIVED")
        failed_agents = self.agent_df.filter(pl.col("status") == "FAILED")

        # Also check for other potential status values
        print("Available statuses:", self.agent_df["status"].unique().to_list())

        if len(arrived_agents) == 0:
            avg_svi_arrived = 0.0
            print("Warning: No agents with ARRIVED status found")
        else:
            avg_svi_arrived = arrived_agents["SVI"].mean()

        if len(failed_agents) == 0:
            avg_svi_failed = 0.0
            print("Warning: No agents with FAILED status found")
        else:
            avg_svi_failed = failed_agents["SVI"].mean()

        print("\n--- SVI vs. Evacuation Outcome Analysis ---")
        print(f"Total agents: {len(self.agent_df)}")
        print(f"Successfully evacuated agents: {len(arrived_agents)}")
        print(f"Failed/trapped agents: {len(failed_agents)}")
        print(f"Average SVI of Successfully Evacuated Agents: {avg_svi_arrived:.3f}")
        print(f"Average SVI of Failed/Trapped Agents: {avg_svi_failed:.3f}")

        # Status breakdown
        status_counts = self.agent_df.group_by("status").agg(pl.len().alias("count"))
        print("\nStatus breakdown:")
        for row in status_counts.iter_rows(named=True):
            print(f"  {row['status']}: {row['count']}")

        print("-" * 40)
        return {"avg_svi_arrived": avg_svi_arrived, "avg_svi_failed": avg_svi_failed}

    @autosave
    def plot_svi_vs_evacuation_time(self, figsize=(10, 6)):
        """Plots SVI against the total evacuation time for successful agents."""
        arrived_agents = self.agent_df.filter(pl.col("status") == "ARRIVED")

        if len(arrived_agents) == 0:
            print("No successfully evacuated agents to plot.")
            # Try alternative status names
            print("Trying alternative status names...")
            alt_statuses = ["arrived", "EVACUATED", "evacuated", "SUCCESS", "success"]
            for alt_status in alt_statuses:
                alt_agents = self.agent_df.filter(pl.col("status") == alt_status)
                if len(alt_agents) > 0:
                    print(f"Found {len(alt_agents)} agents with status '{alt_status}'")
                    arrived_agents = alt_agents
                    break

            if len(arrived_agents) == 0:
                print("No agents found with any success status.")
                return None

        # Convert to pandas for seaborn compatibility
        arrived_pandas = arrived_agents.to_pandas()

        fig, ax = plt.subplots(figsize=figsize)

        # Check if we have valid evacuation time data
        if (
            arrived_pandas["evacuation_time"].isna().all()
            or (arrived_pandas["evacuation_time"] == 0).all()
        ):
            print("Warning: No valid evacuation time data found")
            # Create a simple scatter plot without regression
            ax.scatter(
                arrived_pandas["SVI"],
                arrived_pandas["evacuation_time"],
                alpha=0.6,
                s=50,
            )
        else:
            sns.regplot(
                data=arrived_pandas,
                x="SVI",
                y="evacuation_time",
                scatter_kws={"alpha": 0.4, "s": 50},
                line_kws={"color": "red", "linewidth": 3},
                ax=ax,
            )

        ax.set_title(
            "Social Vulnerability Index vs. Evacuation Time",
            fontsize=16,
            fontweight="bold",
        )
        ax.set_xlabel("SVI (Higher = More Vulnerable)", fontsize=12)
        ax.set_ylabel("Evacuation Time (seconds)", fontsize=12)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)

        plt.tight_layout()
        plt.show()
        return fig

    # --- INSIGHT 2: THE EVACUATION EQUITY GAP ---
    def analyze_equity_gap(self):
        """Analyzes evacuation outcomes by SVI quintile."""
        df = self.agent_df.filter(pl.col("status") == "ARRIVED")

        # Try alternative status if ARRIVED not found
        if len(df) == 0:
            alt_statuses = ["arrived", "EVACUATED", "evacuated", "SUCCESS", "success"]
            for alt_status in alt_statuses:
                df = self.agent_df.filter(pl.col("status") == alt_status)
                if len(df) > 0:
                    break

        if len(df) == 0:
            print("No successfully evacuated agents for equity gap analysis.")
            return pl.DataFrame()

        # Create quintiles
        try:
            df = df.with_columns(
                pl.col("SVI")
                .qcut(5, labels=[f"Q{i}" for i in range(1, 6)])
                .alias("svi_quintile")
            )
        except Exception as e:
            print(f"Error creating quintiles: {e}")
            # Fallback: manual quintiles
            svi_values = df["SVI"].to_numpy()
            quintiles = np.percentile(svi_values, [20, 40, 60, 80])

            def assign_quintile(svi):
                if svi <= quintiles[0]:
                    return "Q1"
                elif svi <= quintiles[1]:
                    return "Q2"
                elif svi <= quintiles[2]:
                    return "Q3"
                elif svi <= quintiles[3]:
                    return "Q4"
                else:
                    return "Q5"

            df = df.with_columns(
                pl.col("SVI")
                .map_elements(assign_quintile, return_dtype=pl.String)
                .alias("svi_quintile")
            )

        # Group by quintile and calculate statistics
        equity_gap_stats = (
            df.group_by("svi_quintile")
            .agg(
                [
                    pl.col("evacuation_time").mean().alias("mean_evacuation_time"),
                    pl.col("evacuation_time").median().alias("median_evacuation_time"),
                    pl.len().alias("count"),
                ]
            )
            .sort("svi_quintile")
        )

        print("\n--- Evacuation Equity Gap Analysis (by SVI Quintile) ---")
        print(equity_gap_stats)
        print("-" * 40)
        return equity_gap_stats

    @autosave
    def plot_equity_gap(self, figsize=(10, 6)):
        """Creates a bar plot showing the average evacuation time by SVI quintile."""
        stats = self.analyze_equity_gap()

        if len(stats) == 0:
            return None

        # Convert to pandas for plotting
        stats_pandas = stats.to_pandas()

        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.bar(
            stats_pandas["svi_quintile"],
            stats_pandas["mean_evacuation_time"],
            color=sns.color_palette("viridis", len(stats_pandas)),
            edgecolor="black",
        )

        ax.set_title("Evacuation Equity Gap", fontsize=16, fontweight="bold")
        ax.set_xlabel(
            "SVI Quintile (Q1 = Least Vulnerable, Q5 = Most Vulnerable)", fontsize=12
        )
        ax.set_ylabel("Average Evacuation Time (seconds)", fontsize=12)
        ax.tick_params(axis="x", rotation=0)
        ax.grid(axis="y", linestyle="--", linewidth=0.5)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.1f}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        plt.show()
        return fig

    # --- INSIGHT 3: GEOSPATIAL BOTTLENECK ANALYSIS ---
    @autosave
    def plot_bottleneck_map(self, G_drive=None, figsize=(15, 15)):
        """Plots the drive network, highlighting bottlenecks colored by the SVI of stuck agents."""
        if len(self.bottleneck_df) == 0:
            print("No bottleneck data to plot.")
            return None

        if G_drive is None and self.model is not None:
            # Try to get graph from model
            if hasattr(self.model, "graphs") and "CAR" in self.model.graphs:
                G_drive = self.model.graphs["CAR"]
            else:
                print("No drive network graph available.")
                return None
        elif G_drive is None:
            print("No drive network graph provided.")
            return None

        # Aggregate bottleneck data: find the maximum congestion for each edge
        edge_agg = self.bottleneck_df.group_by("edge_nodes").agg(
            [
                pl.col("congestion_index").max().alias("max_congestion"),
                pl.col("avg_svi_stuck").mean().alias("avg_svi_stuck"),
            ]
        )

        edge_colors = {}
        edge_tuples = []

        for row in edge_agg.iter_rows(named=True):
            edge_str = row["edge_nodes"]
            avg_svi = row["avg_svi_stuck"]

            # Parse edge string to tuple
            try:
                # Handle different formats: "(node1, node2)" or "node1, node2"
                edge_clean = edge_str.strip("()").split(", ")
                if len(edge_clean) == 2:
                    edge_tuple = (int(edge_clean[0]), int(edge_clean[1]))
                    edge_tuples.append(edge_tuple)

                    # Use plasma colormap for SVI (assuming SVI is 0-1)
                    svi_color = cm.get_cmap("plasma")(min(max(avg_svi, 0), 1))
                    edge_colors[edge_tuple] = svi_color
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse edge {edge_str}: {e}")
                continue

        if not edge_colors:
            print("No valid bottleneck edges found to plot.")
            return None

        # Get the colors and widths for edges
        ec = []
        ew = []

        for edge in G_drive.edges():
            if edge in edge_colors:
                ec.append(edge_colors[edge])
                ew.append(5)  # Thick line for bottlenecks
            else:
                ec.append("lightgray")
                ew.append(0.5)  # Thin line for normal edges

        print(f"Plotting map with {len(edge_colors)} bottlenecked edges.")

        fig, ax = ox.plot_graph(
            G_drive,
            edge_color=ec,
            edge_linewidth=ew,
            node_size=0,
            figsize=figsize,
            bgcolor="#FFFFFF",
            show=False,
            close=False,
        )

        # Add a colorbar for SVI
        sm = plt.cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", pad=0.02, shrink=0.5)
        cbar.set_label("Average SVI of Agents in Bottleneck", fontsize=12)

        ax.set_title("Geospatial Bottleneck Analysis", fontsize=18, fontweight="bold")
        plt.tight_layout()
        plt.show()
        return fig

    # --- ADDITIONAL ANALYTICS FOR AGENTPY MODEL ---
    @autosave
    def plot_agent_status_distribution(self, figsize=(10, 6)):
        """Plot distribution of agent final statuses."""
        status_counts = self.agent_df.group_by("status").agg(pl.len().alias("count"))
        status_pandas = status_counts.to_pandas()

        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.bar(
            status_pandas["status"],
            status_pandas["count"],
            color=sns.color_palette("Set2", len(status_pandas)),
        )

        ax.set_title("Agent Status Distribution", fontsize=16, fontweight="bold")
        ax.set_xlabel("Status", fontsize=12)
        ax.set_ylabel("Number of Agents", fontsize=12)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
            )

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        return fig

    @autosave
    def plot_transportation_mode_analysis(self, figsize=(12, 8)):
        """Analyze evacuation success by transportation mode."""
        mode_status = (
            self.agent_df.group_by(["main_mode", "status"])
            .agg(pl.len().alias("count"))
            .pivot(index="main_mode", columns="status", values="count")
            .fill_null(0)
        )

        mode_pandas = mode_status.to_pandas()

        fig, ax = plt.subplots(figsize=figsize)
        mode_pandas.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            color=sns.color_palette("Set3", len(mode_pandas.columns)),
        )

        ax.set_title(
            "Evacuation Outcomes by Transportation Mode", fontsize=16, fontweight="bold"
        )
        ax.set_xlabel("Transportation Mode", fontsize=12)
        ax.set_ylabel("Number of Agents", fontsize=12)
        ax.legend(title="Status", bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        return fig

    def generate_summary_report(self):
        """Generate a comprehensive summary report."""
        print("\n" + "=" * 50)
        print("EVACUATION SIMULATION SUMMARY REPORT")
        print("=" * 50)

        # Basic statistics
        total_agents = len(self.agent_df)
        status_counts = self.agent_df.group_by("status").agg(pl.count().alias("count"))

        print(f"\nTotal Agents: {total_agents}")
        print("\nStatus Distribution:")
        for row in status_counts.iter_rows(named=True):
            status = row["status"]
            count = row["count"]
            percentage = (count / total_agents) * 100
            print(f"  {status}: {count} ({percentage:.1f}%)")

        # SVI analysis
        avg_svi = self.agent_df["SVI"].mean()
        print(f"\nAverage SVI: {avg_svi:.3f}")

        # Mode distribution
        mode_counts = self.agent_df.group_by("main_mode").agg(pl.count().alias("count"))
        print("\nTransportation Mode Distribution:")
        for row in mode_counts.iter_rows(named=True):
            mode = row["main_mode"]
            count = row["count"]
            percentage = (count / total_agents) * 100
            print(f"  {mode}: {count} ({percentage:.1f}%)")

        # Success analysis
        self.analyze_svi_vs_outcome()

        print("=" * 50)


def _save_figure_static(fig, func_name: str, **save_kwargs):
    """Static method for saving figures when not in class context."""
    out_dir = save_kwargs.get("folder", "plots")
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    fname = save_kwargs.get("filename", f"{func_name}.png")
    dpi = save_kwargs.get("dpi", 300)
    fig.savefig(f"{out_dir}/{fname}", dpi=dpi, bbox_inches="tight")
    print(f"Plot saved to {out_dir}/{fname}")
