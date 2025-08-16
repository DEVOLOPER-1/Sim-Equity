# FILE: simulation_analytics.py
# -----------------------------
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

    def _process_bottlenecks(self) -> pl.DataFrame:
        all_logs = []
        for index, row in self.raw_model_df.iter_rows():
            for log_entry in row["bottlenecks"]:
                all_logs.append(log_entry)
        return pl.DataFrame(all_logs)

    # --- INSIGHT 1: SVI vs. EVACUATION OUTCOME ---
    def analyze_svi_vs_outcome(self):
        """Calculates and prints key stats about SVI and evacuation success."""
        arrived_agents = self.agent_df[self.agent_df["status"] == "arrived"]
        failed_agents = self.agent_df[self.agent_df["status"] == "failed"]

        avg_svi_arrived = arrived_agents["SVI"].mean()
        avg_svi_failed = failed_agents["SVI"].mean()

        print("\n--- SVI vs. Evacuation Outcome Analysis ---")
        print(f"Average SVI of Successfully Evacuated Agents: {avg_svi_arrived:.3f}")
        print(f"Average SVI of Failed/Trapped Agents: {avg_svi_failed:.3f}")
        print("-" * 40)
        return {"avg_svi_arrived": avg_svi_arrived, "avg_svi_failed": avg_svi_failed}

    def plot_svi_vs_evacuation_time(self, figsize=(10, 6)):
        """Plots SVI against the total evacuation time for successful agents."""
        arrived_agents = self.agent_df[self.agent_df["status"] == "arrived"].clone()

        plt.figure(figsize=figsize)
        sns.regplot(
            data=arrived_agents,
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
        df = self.agent_df[self.agent_df["status"] == "arrived"].clone()
        df = df.with_columns(
            pl.col("SVI")
            .qcut(5, labels=[f"Q{i}" for i in range(1, 6)])
            .alias("svi_quintile")
        )

        equity_gap_stats = df.group_by(["svi_quintile", "evacuation_time"]).agg(
            ["mean", "median", "count"]
        )

        print("\n--- Evacuation Equity Gap Analysis (by SVI Quintile) ---")
        print(equity_gap_stats)
        print("-" * 40)
        return equity_gap_stats

    def plot_equity_gap(self, figsize=(10, 6)):
        """Creates a bar plot showing the average evacuation time by SVI quintile."""
        stats = self.analyze_equity_gap()

        plt.figure(figsize=figsize)
        stats["mean"].plot(
            kind="bar", color=sns.color_palette("viridis", 5), edgecolor="black"
        )
        plt.title("Evacuation Equity Gap", fontsize=16, fontweight="bold")
        plt.xlabel(
            "SVI Quintile (Q1 = Least Vulnerable, Q5 = Most Vulnerable)", fontsize=12
        )
        plt.ylabel("Average Evacuation Time (seconds)", fontsize=12)
        plt.xticks(rotation=0)
        plt.grid(axis="y", linestyle="--", linewidth=0.5)
        plt.show()

    # --- INSIGHT 3: GEOSPATIAL BOTTLENECK ANALYSIS ---
    def plot_bottleneck_map(self, G_drive, figsize=(15, 15)):
        """Plots the drive network, highlighting bottlenecks colored by the SVI of stuck agents."""
        if self.bottleneck_df.is_empty():
            print("No bottleneck data to plot.")
            return

        # Aggregate bottleneck data: find the *worst* state for each edge
        edge_agg = self.bottleneck_df.select(
            self.bottleneck_df.group_by(["edge_nodes", "congestion_index"]).idxmax()
        )

        edge_colors = {}
        for _, row in edge_agg.iter_rows():
            edge = row["edge_nodes"]
            svi_color = plt.cm.plasma(row["avg_svi_stuck"])  # SVI is already 0-1
            edge_colors[edge] = svi_color

        # Get the colors for the edges we want to plot
        ec = [edge_colors.get(edge, "gray") for edge in G_drive.edges()]
        ew = [5 if edge in edge_colors else 0.5 for edge in G_drive.edges()]

        print(f"Plotting map with {len(edge_colors)} bottlenecked edges.")
        fig, ax = ox.plot_graph(
            G_drive,
            edge_color=ec,
            edge_linewidth=ew,
            node_size=0,
            figsize=figsize,
            bgcolor="#FFFFFF",
            show=False,
            close=True,
        )

        # Add a colorbar for SVI
        sm = plt.cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", pad=0.02, shrink=0.5)
        cbar.set_label("Average SVI of Agents in Bottleneck", fontsize=12)

        ax.set_title("Geospatial Bottleneck Analysis", fontsize=18, fontweight="bold")
        plt.show()
