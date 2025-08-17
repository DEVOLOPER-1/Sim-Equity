# FILE: simulation_analytics.py
# -----------------------------
import pathlib
from functools import wraps
from typing import Any, Callable

import matplotlib.pyplot as plt
import osmnx as ox
import polars as pl
import seaborn as sns


class SimulationAnalytics:
    def __init__(self, model_data: pl.DataFrame, agent_data: pl.DataFrame):
        """
        Initialize the analytics suite with data from a completed Mesa simulation.

        Args:
            model_data: DataFrame from model_reporters (e.g., bottlenecks).
            agent_data: DataFrame from agent_reporters (e.g., final agent states).
        """
        self.raw_model_df = model_data
        self.agent_df = agent_data
        print(f"Analytics initialized with {len(self.agent_df)} agent records.")

        # Process the raw bottleneck log into a more usable format
        self.bottleneck_df = self._process_bottlenecks()

    def autosave(func: Callable) -> Callable:
        """
        Decorator for plot methods to add optional saving behaviour.

        Usage: wrapped_plot(self, ..., save=True, save_kwargs={"filename":"my.png"})
        """

        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Extract save kwargs (remove them before calling original function)
            save = bool(kwargs.pop("save", True))
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

                # Call instance save_figure if available
                if fig is not None:
                    # assume first arg is self
                    if len(args) > 0:
                        self_obj = args[0]
                        # if object has save_figure method, use it; else use fig.savefig directly
                        if hasattr(self_obj, "save_figure") and callable(
                            getattr(self_obj, "save_figure")
                        ):
                            self_obj.save_figure(fig=fig, **save_kwargs)
                        else:
                            # fallback direct save into 'plots' folder
                            out_dir = save_kwargs.get("folder", "plots")
                            pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
                            fname = save_kwargs.get("filename", f"{func.__name__}.png")
                            dpi = save_kwargs.get("dpi", 600)
                            fig.savefig(
                                f"{out_dir}/{fname}", dpi=dpi, bbox_inches="tight"
                            )
                    else:
                        # No self; just save fig directly
                        out_dir = save_kwargs.get("folder", "plots")
                        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
                        fname = save_kwargs.get("filename", f"{func.__name__}.png")
                        dpi = save_kwargs.get("dpi", 600)
                        fig.savefig(f"{out_dir}/{fname}", dpi=dpi, bbox_inches="tight")

            # Return original return value to preserve behaviour
            return result

        return wrapper

    def _process_bottlenecks(self) -> pl.DataFrame:
        """Process raw bottleneck data into a structured DataFrame."""
        all_logs = []
        for row in self.raw_model_df.iter_rows(named=True):
            bottlenecks = row.get("bottlenecks", [])
            if bottlenecks:  # Check if bottlenecks exist and is not empty
                for log_entry in bottlenecks:
                    all_logs.append(log_entry)

        if not all_logs:
            # Return empty DataFrame with expected schema if no bottlenecks
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
        arrived_agents = self.agent_df.filter(pl.col("status") == "arrived")
        failed_agents = self.agent_df.filter(pl.col("status") == "failed")

        if len(arrived_agents) == 0:
            avg_svi_arrived = 0.0
        else:
            avg_svi_arrived = arrived_agents["SVI"].mean()

        if len(failed_agents) == 0:
            avg_svi_failed = 0.0
        else:
            avg_svi_failed = failed_agents["SVI"].mean()

        print("\n--- SVI vs. Evacuation Outcome Analysis ---")
        print(f"Total agents: {len(self.agent_df)}")
        print(f"Successfully evacuated agents: {len(arrived_agents)}")
        print(f"Failed/trapped agents: {len(failed_agents)}")
        print(f"Average SVI of Successfully Evacuated Agents: {avg_svi_arrived:.3f}")
        print(f"Average SVI of Failed/Trapped Agents: {avg_svi_failed:.3f}")
        print("-" * 40)
        return {"avg_svi_arrived": avg_svi_arrived, "avg_svi_failed": avg_svi_failed}

    @autosave
    def plot_svi_vs_evacuation_time(self, figsize=(10, 6)):
        """Plots SVI against the total evacuation time for successful agents."""
        arrived_agents = self.agent_df.filter(pl.col("status") == "arrived")

        if len(arrived_agents) == 0:
            print("No successfully evacuated agents to plot.")
            return

        # Convert to pandas for seaborn compatibility
        arrived_pandas = arrived_agents.to_pandas()

        plt.figure(figsize=figsize)
        sns.regplot(
            data=arrived_pandas,
            x="SVI",
            y="evacuation_time",
            scatter_kws={"alpha": 0.4, "s": 50},
            line_kws={"color": "red", "linewidth": 3},
        )
        plt.title(
            "Social Vulnerability Index vs. Evacuation Time",
            fontsize=16,
            fontweight="bold",
        )
        plt.xlabel("SVI (Higher = More Vulnerable)", fontsize=12)
        plt.ylabel("Evacuation Time (seconds)", fontsize=12)
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.show()

    # --- INSIGHT 2: THE EVACUATION EQUITY GAP ---
    def analyze_equity_gap(self):
        """Analyzes evacuation outcomes by SVI quintile."""
        df = self.agent_df.filter(pl.col("status") == "arrived")

        if len(df) == 0:
            print("No successfully evacuated agents for equity gap analysis.")
            return pl.DataFrame()

        # Create quintiles
        df = df.with_columns(
            pl.col("SVI")
            .qcut(5, labels=[f"Q{i}" for i in range(1, 6)])
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
            return

        # Convert to pandas for plotting
        stats_pandas = stats.to_pandas()

        plt.figure(figsize=figsize)
        bars = plt.bar(
            stats_pandas["svi_quintile"],
            stats_pandas["mean_evacuation_time"],
            color=sns.color_palette("viridis", 5),
            edgecolor="black",
        )

        plt.title("Evacuation Equity Gap", fontsize=16, fontweight="bold")
        plt.xlabel(
            "SVI Quintile (Q1 = Least Vulnerable, Q5 = Most Vulnerable)", fontsize=12
        )
        plt.ylabel("Average Evacuation Time (seconds)", fontsize=12)
        plt.xticks(rotation=0)
        plt.grid(axis="y", linestyle="--", linewidth=0.5)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.1f}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        plt.show()

    # --- INSIGHT 3: GEOSPATIAL BOTTLENECK ANALYSIS ---
    @autosave
    def plot_bottleneck_map(self, G_drive, figsize=(15, 15)):
        """Plots the drive network, highlighting bottlenecks colored by the SVI of stuck agents."""
        if len(self.bottleneck_df) == 0:
            print("No bottleneck data to plot.")
            return

        # Aggregate bottleneck data: find the maximum congestion for each edge
        edge_agg = self.bottleneck_df.group_by("edge_nodes").agg(
            [
                pl.col("congestion_index").max().alias("max_congestion"),
                pl.col("avg_svi_stuck")
                .mean()
                .alias("avg_svi_stuck"),  # Average SVI across all bottleneck events
            ]
        )

        edge_colors = {}
        edge_tuples = []

        for row in edge_agg.iter_rows(named=True):
            edge_str = row["edge_nodes"]
            avg_svi = row["avg_svi_stuck"]

            # Parse edge string to tuple (assuming format like "(node1, node2)")
            try:
                # Remove parentheses and split by comma
                edge_clean = edge_str.strip("()").split(", ")
                if len(edge_clean) == 2:
                    edge_tuple = (int(edge_clean[0]), int(edge_clean[1]))
                    edge_tuples.append(edge_tuple)

                    # Use plasma colormap for SVI (assuming SVI is 0-1)
                    svi_color = plt.cm.plasma(min(max(avg_svi, 0), 1))
                    edge_colors[edge_tuple] = svi_color
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse edge {edge_str}: {e}")
                continue

        if not edge_colors:
            print("No valid bottleneck edges found to plot.")
            return

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
