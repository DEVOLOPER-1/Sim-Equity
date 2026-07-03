# %%
import warnings

warnings.filterwarnings("ignore")

# %%
# IMPORTS AND GLOBAL SETTINGS
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import polars as pl
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

_ROOT = Path(__file__).parent.parent.parent

# ---------------------------
# Global style / constants
# ---------------------------
FIGSIZE = (6.685, 6.25)
DPI = 300

# Base font sizes
BASE_FONTSIZE = 17
TITLE_FONTSIZE = 20
TICK_FONTSIZE = 17
ANNOT_FONTSIZE = 17
COLORBAR_FONTSIZE = 17
MATH_FONTSIZE = 18

MARKER_SIZE = 36
ALPHA = 0.6
GRID_STYLE = {"linestyle": "--", "alpha": 0.3, "linewidth": 0.8}

# Define colors for consistency
primary_color = "#2E86AB"
secondary_color = "#A23B72"
accent_color = "#F18F01"
grid_color = "#E5E5E5"
success_color = "#27AE60"
warning_color = "#F39C12"
danger_color = "#E74C3C"
neutral_color = "#95A5A6"

# Vulnerability level colors
VULNERABILITY_COLORS = {
    "low": "#27AE60",
    "moderate": "#F39C12",
    "high": "#E67E22",
    "very_high": "#E74C3C",
}

# Apply consistent rcParams
plt.rcParams.update(
    {
        "figure.figsize": FIGSIZE,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "font.family": "DejaVu Sans",
        "font.size": BASE_FONTSIZE,
        "axes.titlesize": TITLE_FONTSIZE,
        "axes.labelsize": BASE_FONTSIZE,
        "xtick.labelsize": TICK_FONTSIZE,
        "ytick.labelsize": TICK_FONTSIZE,
        "legend.fontsize": BASE_FONTSIZE,
        "mathtext.fontset": "dejavusans",
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.2,
        "grid.linewidth": 0.5,
        "lines.linewidth": 2.0,
        "patch.linewidth": 1.0,
        "boxplot.flierprops.markersize": 4,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "text.usetex": False,
    }
)


# Utility functions for consistent plotting
def setup_academic_grid(ax, axis="both"):
    """Apply consistent grid styling to axes"""
    ax.grid(True, axis=axis, linestyle=":", linewidth=0.5, color=grid_color, alpha=0.7)


def add_vulnerability_zones(ax, axis="x", alpha=0.1):
    """Add vulnerability level background zones for SVI plots"""
    if axis == "x":
        ax.axvspan(
            0,
            0.25,
            alpha=alpha,
            color=VULNERABILITY_COLORS["low"],
            label="Low Vulnerability",
        )
        ax.axvspan(
            0.25,
            0.50,
            alpha=alpha,
            color=VULNERABILITY_COLORS["moderate"],
            label="Moderate Vulnerability",
        )
        ax.axvspan(
            0.50,
            0.75,
            alpha=alpha,
            color=VULNERABILITY_COLORS["high"],
            label="High Vulnerability",
        )
        ax.axvspan(
            0.75,
            1,
            alpha=alpha,
            color=VULNERABILITY_COLORS["very_high"],
            label="Very High Vulnerability",
        )
    elif axis == "y":
        ax.axhspan(0, 0.25, alpha=alpha, color=VULNERABILITY_COLORS["low"])
        ax.axhspan(0.25, 0.50, alpha=alpha, color=VULNERABILITY_COLORS["moderate"])
        ax.axhspan(0.50, 0.75, alpha=alpha, color=VULNERABILITY_COLORS["high"])
        ax.axhspan(0.75, 1, alpha=alpha, color=VULNERABILITY_COLORS["very_high"])


def savefig_standard(fname, dpi=None):
    """Save figure with consistent academic formatting"""
    if dpi is None:
        dpi = DPI
    plt.tight_layout()
    plt.savefig(
        fname,
        bbox_inches="tight",
        dpi=dpi,
        facecolor="white",
        edgecolor="none",
        transparent=False,
    )
    print(f"Saved: {fname}")


# %%
BASE_SAVE_DIR = str(_ROOT / "outputs" / "figures" / "evacuation_analytics")


# %%
def load_agent_traces(
    agent_summary: pl.DataFrame,
) -> Tuple[pl.DataFrame, Dict[str, list]]:
    """
    Load all agent traces from CSV files with robust coordinate handling.
    Extract original agent ID from filename and integrate with SVI data.

    Returns:
        Dictionary mapping agent IDs to their coordinate trajectories
    """

    print("Loading agent traces...")
    agent_traces = {}
    traces_loaded = 0
    expanded_agent_data = []

    try:
        trace_dir = str(_ROOT / "outputs" / "agent_states" / "simulation_outcomes") + "/agents_traces"
        if not os.path.exists(trace_dir):
            print(f"Trace directory {trace_dir} does not exist.")
            return agent_summary, agent_traces

        # Create a mapping from original agent ID to agent data
        agent_data_map = {}
        for row in agent_summary.iter_rows(named=True):
            agent_data_map[row["agent_id"]] = row

        # Get all trace files
        trace_files = [f for f in os.listdir(trace_dir) if f.endswith(".csv")]

        for trace_file in trace_files:
            try:
                # Extract original agent ID from filename
                filename_without_ext = trace_file.replace(".csv", "")
                parts = filename_without_ext.split("_")
                original_agent_id = "_".join(parts[:-1])  # Remove the trailing index

                # Get agent data from the mapping
                if original_agent_id not in agent_data_map:
                    print(
                        f"Original agent ID {original_agent_id} not found in summary data for trace file {trace_file}"
                    )
                    continue

                agent_data = agent_data_map[original_agent_id]

                trace_path = os.path.join(trace_dir, trace_file)
                df = pl.read_csv(trace_path)

                if df.is_empty():
                    continue

                # Extract coordinates - simulation uses y=lat, x=lon
                coordinates = []
                for row in df.iter_rows(named=True):
                    if row["y"] is not None and row["x"] is not None:
                        coordinates.append(
                            [float(row["y"]), float(row["x"])]
                        )  # [lat, lon]

                if len(coordinates) >= 2:
                    # Use the trace filename as the unique agent ID
                    unique_agent_id = filename_without_ext
                    agent_traces[unique_agent_id] = coordinates

                    # Create expanded agent data with the unique ID
                    expanded_agent = dict(agent_data)
                    expanded_agent["agent_id"] = unique_agent_id
                    expanded_agent["original_agent_id"] = original_agent_id
                    expanded_agent_data.append(expanded_agent)

                    traces_loaded += 1

            except Exception as e:
                print(f"Error loading trace for file {trace_file}: {e}")

        # Create a new agent summary with the expanded data
        if expanded_agent_data:
            expanded_agent_summary = pl.DataFrame(expanded_agent_data)
        else:
            expanded_agent_summary = agent_summary

        print(
            f"Successfully loaded traces for {traces_loaded} out of {len(trace_files)} trace files"
        )

    except Exception as e:
        print(f"Error processing agent traces: {e}")
        expanded_agent_summary = agent_summary

    return expanded_agent_summary, agent_traces


def load_simulation_data() -> Tuple[pl.DataFrame, pl.DataFrame, Dict[str, list]]:
    """Load all simulation outcome data"""
    print("Loading simulation data...")

    # Load agent summary statistics
    agent_summary = pl.read_csv(str(_ROOT / "outputs" / "agent_states" / "simulation_outcomes") + "/Agents_Statistics_Trial.csv")

    # Load journey segment details if available
    journey_segments_path = str(_ROOT / "outputs" / "agent_states" / "simulation_outcomes") + "/Journey_Segments_Detail.csv"
    if os.path.exists(journey_segments_path):
        journey_segments = pl.read_csv(journey_segments_path)
    else:
        journey_segments = pl.DataFrame()

    # Load agent traces and get expanded agent summary
    agent_summary, agent_traces = load_agent_traces(agent_summary)

    return agent_summary, journey_segments, agent_traces


def preprocess_data(
    agent_summary: pl.DataFrame, journey_segments: pl.DataFrame
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Clean and preprocess the data for analysis

    Args:
        agent_summary: Raw agent summary data
        journey_segments: Raw journey segments data

    Returns:
        Preprocessed agent summary and journey segments
    """
    # Convert SVI to categorical vulnerability levels
    agent_summary = agent_summary.with_columns(
        pl.when(pl.col("svi") <= 0.25)
        .then(pl.lit("low"))
        .when(pl.col("svi") <= 0.5)
        .then(pl.lit("moderate"))
        .when(pl.col("svi") <= 0.75)
        .then(pl.lit("high"))
        .otherwise(pl.lit("very_high"))
        .alias("vulnerability_level")
    )

    # Ensure evacuation time is in minutes
    if "evacuation_time_seconds" in agent_summary.columns:
        agent_summary = agent_summary.with_columns(
            (pl.col("evacuation_time_seconds") / 60).alias("evacuation_time_minutes")
        )

    # Categorize success
    agent_summary = agent_summary.with_columns(
        (pl.col("status") == "ARRIVED").alias("success")
    )

    # Identify public transport users more accurately
    if "used_public_transport" not in agent_summary.columns:
        # Check if agent has journey segments
        if "agent_id" in journey_segments.columns and not journey_segments.is_empty():
            # Use the original agent ID for matching
            if "original_agent_id" in agent_summary.columns:
                pt_users = journey_segments["agent_id"].unique()
                agent_summary = agent_summary.with_columns(
                    pl.col("original_agent_id")
                    .is_in(pt_users)
                    .alias("used_public_transport")
                )
            else:
                pt_users = journey_segments["agent_id"].unique()
                agent_summary = agent_summary.with_columns(
                    pl.col("agent_id").is_in(pt_users).alias("used_public_transport")
                )
        else:
            agent_summary = agent_summary.with_columns(
                pl.lit(False).alias("used_public_transport")
            )

    return agent_summary, journey_segments


# Load and preprocess data
agent_summary, journey_segments, agent_traces = load_simulation_data()
agent_summary, journey_segments = preprocess_data(agent_summary, journey_segments)

# %%
# INDIVIDUAL VISUALIZATION FUNCTIONS


def plot_mode_distribution(
    agent_summary: pl.DataFrame, save_path: Optional[str] = None
) -> plt.Figure:
    """Plot transport mode distribution as a separate figure"""
    df = agent_summary.to_pandas()

    fig, ax = plt.subplots(figsize=FIGSIZE)

    if "final_mode" in df.columns:
        mode_counts = df["final_mode"].value_counts().sort_values()
        colors = sns.color_palette("muted", len(mode_counts))
        ax.barh(
            mode_counts.index,
            mode_counts.values,
            color=colors,
            edgecolor="k",
            linewidth=0.6,
        )
        ax.set_title("Transport Mode Distribution", fontsize=TITLE_FONTSIZE)
        ax.set_xlabel("Number of agents")

        # Annotate counts
        for i, (label, val) in enumerate(mode_counts.items()):
            ax.text(
                val + max(mode_counts.values) * 0.01,
                i,
                f"{val:,}",
                va="center",
                fontsize=BASE_FONTSIZE - 2,
            )
    else:
        ax.text(
            0.5,
            0.5,
            "No 'final_mode' column",
            ha="center",
            va="center",
            fontsize=BASE_FONTSIZE,
        )

    setup_academic_grid(ax, axis="x")
    plt.tight_layout()

    if save_path:
        savefig_standard(save_path)

    return fig


def plot_success_rate_by_mode(
    agent_summary: pl.DataFrame, save_path: Optional[str] = None
) -> plt.Figure:
    """Plot success rate by transport mode as a separate figure"""
    df = agent_summary.to_pandas()

    fig, ax = plt.subplots(figsize=FIGSIZE)

    if "final_mode" in df.columns and "success" in df.columns:
        success_by_mode = df.groupby("final_mode")["success"].agg(["mean", "count"])
        success_by_mode["mean_pct"] = success_by_mode["mean"] * 100
        success_by_mode = success_by_mode.sort_values("mean_pct", ascending=False)

        bars = ax.bar(
            success_by_mode.index,
            success_by_mode["mean_pct"],
            edgecolor="k",
            linewidth=0.6,
            alpha=0.9,
            color=primary_color,
        )

        ax.set_title("Success Rate by Transport Mode", fontsize=TITLE_FONTSIZE)
        ax.set_ylabel("Success Rate (%)")
        ax.tick_params(axis="x", rotation=45)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100.0))
        ax.set_ylim(0, 100)

        # Annotate each bar with percentage and count
        for i, (mode, row) in enumerate(success_by_mode.iterrows()):
            pct = row["mean_pct"]
            n = int(row["count"])
            ax.text(
                i,
                pct + (ax.get_ylim()[1] * 0.02),
                f"{pct:.1f}%",
                ha="center",
                va="bottom",
                fontsize=BASE_FONTSIZE - 2,
                fontweight="bold",
            )
    else:
        ax.text(
            0.5,
            0.5,
            "Need 'final_mode' and 'success' columns",
            ha="center",
            va="center",
            fontsize=BASE_FONTSIZE,
        )

    setup_academic_grid(ax, axis="y")
    plt.tight_layout()

    if save_path:
        savefig_standard(save_path)

    return fig


def plot_evacuation_time_by_mode(
    agent_summary: pl.DataFrame, save_path: Optional[str] = None
) -> plt.Figure:
    """Plot evacuation time by mode as a separate figure"""
    df = agent_summary.to_pandas()

    fig, ax = plt.subplots(figsize=FIGSIZE)

    if (
        "evacuation_time_minutes" in df.columns
        and "success" in df.columns
        and "final_mode" in df.columns
    ):

        successful = df[df["success"]].copy()
        order = (
            successful.groupby("final_mode")["evacuation_time_minutes"]
            .median()
            .sort_values()
            .index.tolist()
        )

        sns.boxplot(
            data=successful,
            x="final_mode",
            y="evacuation_time_minutes",
            order=order,
            ax=ax,
            showfliers=True,
            linewidth=1.2,
            palette="pastel",
        )

        ax.set_title(
            "Evacuation Time by Mode (Successful Agents)", fontsize=TITLE_FONTSIZE
        )
        ax.set_xlabel("")
        ax.set_ylabel("Evacuation time (minutes)")
        ax.tick_params(axis="x", rotation=45)
        setup_academic_grid(ax, axis="y")

        # Compute group means and overlay
        means_series = (
            successful.groupby("final_mode")["evacuation_time_minutes"]
            .mean()
            .reindex(order)
        )
        means = means_series.values
        x_positions = np.arange(len(order))

        ax.scatter(
            x_positions,
            means,
            marker="D",
            s=MARKER_SIZE,
            edgecolor=primary_color,
            facecolor="white",
            linewidth=1.5,
            zorder=10,
        )

        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        for i, m in enumerate(means):
            ax.text(
                x_positions[i],
                m + y_range * 0.03,
                f"{m:.1f} min",
                ha="center",
                va="bottom",
                fontsize=BASE_FONTSIZE - 2,
                fontweight="bold",
                color=primary_color,
            )
    else:
        ax.text(
            0.5,
            0.5,
            "Need 'evacuation_time_minutes', 'success' and 'final_mode' columns",
            ha="center",
            va="center",
            fontsize=BASE_FONTSIZE,
        )

    plt.tight_layout()

    if save_path:
        savefig_standard(save_path)

    return fig


def plot_destination_types(
    agent_summary: pl.DataFrame, save_path: Optional[str] = None
) -> plt.Figure:
    """Plot destination types as a separate figure"""
    df = agent_summary.to_pandas()

    fig, ax = plt.subplots(figsize=FIGSIZE)

    if "target_destination_type" in df.columns:
        dest_counts = df["target_destination_type"].value_counts().sort_values()
        ax.bar(
            dest_counts.index,
            dest_counts.values,
            color=neutral_color,
            edgecolor="k",
            linewidth=0.6,
        )
        ax.set_title("Destination Types", fontsize=TITLE_FONTSIZE)
        ax.tick_params(axis="x", rotation=45)
        setup_academic_grid(ax, axis="x")

        # Annotate counts
        for i, (label, val) in enumerate(dest_counts.items()):
            ax.text(
                i,
                val + max(dest_counts.values) * 0.01,
                f"{val:,}",
                ha="center",
                va="bottom",
                fontsize=BASE_FONTSIZE - 2,
            )
    else:
        ax.text(
            0.5,
            0.5,
            "No 'target_destination_type' column",
            ha="center",
            va="center",
            fontsize=BASE_FONTSIZE,
        )

    plt.tight_layout()

    if save_path:
        savefig_standard(save_path)

    return fig


def plot_mode_by_vulnerability(
    agent_summary: pl.DataFrame, save_path: Optional[str] = None
) -> plt.Figure:
    """Plot transport mode by vulnerability level as a separate figure"""
    df = agent_summary.to_pandas()
    vuln_order = ("low", "moderate", "high", "very_high")

    fig, ax = plt.subplots(figsize=FIGSIZE)

    if "final_mode" in df.columns:
        df["vulnerability_level"] = pd.Categorical(
            df["vulnerability_level"], categories=vuln_order, ordered=True
        )

        count_df = (
            df.groupby(["vulnerability_level", "final_mode"])
            .size()
            .reset_index(name="count")
            .pivot(index="vulnerability_level", columns="final_mode", values="count")
            .fillna(0)
            .reindex(vuln_order)
        )

        long = count_df.reset_index().melt(
            id_vars="vulnerability_level", var_name="final_mode", value_name="count"
        )
        total_per_mode = long.groupby("final_mode")["count"].sum()
        modes = total_per_mode[total_per_mode > 0].index.tolist()
        palette = sns.color_palette("muted", len(modes))

        sns.barplot(
            data=long[long["final_mode"].isin(modes)],
            x="vulnerability_level",
            y="count",
            hue="final_mode",
            order=vuln_order,
            hue_order=modes,
            ax=ax,
            palette=palette,
            edgecolor="k",
            linewidth=0.6,
        )

        ax.set_title("Transport Mode by Vulnerability Level", fontsize=TITLE_FONTSIZE)
        ax.set_ylabel("Count")
        ax.set_xlabel("Vulnerability level")
        ax.legend(title="Transport mode", bbox_to_anchor=(1.02, 1), loc="upper left")
        ax.tick_params(axis="x", rotation=20)
        setup_academic_grid(ax, axis="y")

        # Annotate counts
        for p in ax.patches:
            h = p.get_height()
            if h > 0:
                ax.text(
                    p.get_x() + p.get_width() / 2,
                    h + max(1, h * 0.01),
                    f"{int(h):,}",
                    ha="center",
                    va="bottom",
                    fontsize=BASE_FONTSIZE - 3,
                )
    else:
        ax.text(
            0.5,
            0.5,
            "Missing 'final_mode' data",
            ha="center",
            va="center",
            fontsize=BASE_FONTSIZE,
        )

    plt.tight_layout()

    if save_path:
        savefig_standard(save_path)

    return fig


def plot_success_rate_by_vulnerability(
    agent_summary: pl.DataFrame, save_path: Optional[str] = None
) -> plt.Figure:
    """Plot success rate by vulnerability level as a separate figure"""
    df = agent_summary.to_pandas()
    vuln_order = ("low", "moderate", "high", "very_high")

    fig, ax = plt.subplots(figsize=FIGSIZE)

    if "vulnerability_level" in df.columns and "success" in df.columns:
        df["vulnerability_level"] = pd.Categorical(
            df["vulnerability_level"], categories=vuln_order, ordered=True
        )

        success_stats = (
            df.groupby("vulnerability_level")["success"]
            .agg(["mean", "count"])
            .reindex(vuln_order)
            .fillna(0)
        )
        success_stats["mean_pct"] = success_stats["mean"] * 100

        bars = ax.bar(
            success_stats.index.astype(str),
            success_stats["mean_pct"],
            color=primary_color,
            edgecolor="k",
            linewidth=0.6,
            alpha=0.95,
        )

        ax.set_title("Success Rate by Vulnerability Level", fontsize=TITLE_FONTSIZE)
        ax.set_ylabel("Success rate (%)")
        ax.set_xlabel("Vulnerability level")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100.0))
        ax.tick_params(axis="x", rotation=20)
        ax.set_ylim(0, 100)
        setup_academic_grid(ax, axis="y")

        for i, (lvl, row) in enumerate(success_stats.iterrows()):
            pct = row["mean_pct"]
            n = int(row["count"])
            ax.text(
                i,
                pct + (ax.get_ylim()[1] * 0.02),
                f"{pct:.1f}%",
                ha="center",
                va="bottom",
                fontsize=BASE_FONTSIZE - 2,
                fontweight="bold",
            )
    else:
        ax.text(
            0.5,
            0.5,
            "Missing 'vulnerability_level' or 'success' data",
            ha="center",
            va="center",
            fontsize=BASE_FONTSIZE,
        )

    plt.tight_layout()

    if save_path:
        savefig_standard(save_path)

    return fig


def plot_evacuation_time_by_vulnerability(
    agent_summary: pl.DataFrame, save_path: Optional[str] = None
) -> plt.Figure:
    """Plot evacuation time by vulnerability level as a separate figure"""
    df = agent_summary.to_pandas()
    vuln_order = ("low", "moderate", "high", "very_high")

    fig, ax = plt.subplots(figsize=FIGSIZE)

    successful = df[df["success"]].copy()

    if "evacuation_time_minutes" in successful.columns and len(successful) > 0:
        successful["vulnerability_level"] = pd.Categorical(
            successful["vulnerability_level"], categories=vuln_order, ordered=True
        )

        present_lvls = [
            lvl
            for lvl in vuln_order
            if lvl in successful["vulnerability_level"].unique()
        ]

        if present_lvls:
            # Build palette mapping from vulnerability levels
            palette_map = {
                lvl: VULNERABILITY_COLORS.get(lvl, sns.color_palette("pastel")[i])
                for i, lvl in enumerate(present_lvls)
            }

            sns.violinplot(
                data=successful,
                x="vulnerability_level",
                y="evacuation_time_minutes",
                order=present_lvls,
                ax=ax,
                inner="quartile",
                palette=palette_map,
                cut=1,
                linewidth=1.2,
            )

            ax.set_title(
                "Evacuation Time by Vulnerability Level (successful agents)",
                fontsize=TITLE_FONTSIZE,
            )
            ax.set_xlabel("Vulnerability level")
            ax.set_ylabel("Evacuation time (minutes)")
            ax.tick_params(axis="x", rotation=20)
            setup_academic_grid(ax, axis="y")

            # Overlay mean markers and annotate
            means = (
                successful.groupby("vulnerability_level")["evacuation_time_minutes"]
                .mean()
                .reindex(present_lvls)
                .values
            )
            counts = (
                successful.groupby("vulnerability_level")["evacuation_time_minutes"]
                .count()
                .reindex(present_lvls)
                .values
            )
            x_positions = np.arange(len(present_lvls))

            ax.scatter(
                x_positions,
                means,
                marker="D",
                s=MARKER_SIZE,
                edgecolor=primary_color,
                facecolor="white",
                linewidth=1.4,
                zorder=10,
            )

            y_min, y_max = ax.get_ylim()
            y_range = y_max - y_min
            for i, (m, n) in enumerate(zip(means, counts)):
                ax.text(
                    x_positions[i],
                    m + y_range * 0.03,
                    f"{m:.1f} min\n(n={int(n)})",
                    ha="center",
                    va="bottom",
                    fontsize=BASE_FONTSIZE - 3,
                    fontweight="bold",
                    color=primary_color,
                )
        else:
            ax.text(
                0.5,
                0.5,
                "No vulnerability categories found",
                ha="center",
                va="center",
                fontsize=BASE_FONTSIZE,
            )
    else:
        ax.text(
            0.5,
            0.5,
            "Missing 'evacuation_time_minutes' or no successful agents",
            ha="center",
            va="center",
            fontsize=BASE_FONTSIZE,
        )

    plt.tight_layout()

    if save_path:
        savefig_standard(save_path)

    return fig


def plot_distance_by_vulnerability(
    agent_summary: pl.DataFrame, save_path: Optional[str] = None
) -> plt.Figure:
    """Plot destination distance by vulnerability level as a separate figure"""
    df = agent_summary.to_pandas()
    vuln_order = ("low", "moderate", "high", "very_high")

    fig, ax = plt.subplots(figsize=FIGSIZE)

    successful = df[df["success"]].copy()

    if "target_destination_distance_m" in successful.columns and len(successful) > 0:
        successful["vulnerability_level"] = pd.Categorical(
            successful["vulnerability_level"], categories=vuln_order, ordered=True
        )

        present_lvls = [
            lvl
            for lvl in vuln_order
            if lvl in successful["vulnerability_level"].unique()
        ]

        if present_lvls:
            palette_map = {
                lvl: VULNERABILITY_COLORS.get(lvl, sns.color_palette("pastel")[i])
                for i, lvl in enumerate(present_lvls)
            }

            sns.violinplot(
                data=successful,
                x="vulnerability_level",
                y="target_destination_distance_m",
                order=present_lvls,
                ax=ax,
                inner="quartile",
                palette=palette_map,
                cut=1,
                linewidth=1.2,
            )

            ax.set_title(
                "Destination Distance by Vulnerability Level (successful agents)",
                fontsize=TITLE_FONTSIZE,
            )
            ax.set_xlabel("Vulnerability level")
            ax.set_ylabel("Distance (m)")
            ax.tick_params(axis="x", rotation=20)
            setup_academic_grid(ax, axis="y")

            # Overlay mean markers and annotate
            means = (
                successful.groupby("vulnerability_level")[
                    "target_destination_distance_m"
                ]
                .mean()
                .reindex(present_lvls)
                .values
            )
            counts = (
                successful.groupby("vulnerability_level")[
                    "target_destination_distance_m"
                ]
                .count()
                .reindex(present_lvls)
                .values
            )
            x_positions = np.arange(len(present_lvls))

            ax.scatter(
                x_positions,
                means,
                marker="D",
                s=MARKER_SIZE,
                edgecolor=primary_color,
                facecolor="white",
                linewidth=1.4,
                zorder=10,
            )

            y_min, y_max = ax.get_ylim()
            y_range = y_max - y_min
            for i, (m, n) in enumerate(zip(means, counts)):
                # Display km when appropriate
                label = f"{m:.0f} m" if m < 1000 else f"{m / 1000:.2f} km"
                ax.text(
                    x_positions[i],
                    m + y_range * 0.03,
                    f"{label}\n(n={int(n)})",
                    ha="center",
                    va="bottom",
                    fontsize=BASE_FONTSIZE - 3,
                    fontweight="bold",
                    color=primary_color,
                )
        else:
            ax.text(
                0.5,
                0.5,
                "No vulnerability categories found",
                ha="center",
                va="center",
                fontsize=BASE_FONTSIZE,
            )
    else:
        ax.text(
            0.5,
            0.5,
            "Missing 'target_destination_distance_m' or no successful agents",
            ha="center",
            va="center",
            fontsize=BASE_FONTSIZE,
        )

    plt.tight_layout()

    if save_path:
        savefig_standard(save_path)

    return fig


def plot_pt_users_by_vulnerability(
    agent_summary: pl.DataFrame, save_path: Optional[str] = None
) -> plt.Figure:
    """Plot public transport users by vulnerability level as a separate figure"""
    df = agent_summary.to_pandas()
    vuln_order = ("low", "moderate", "high", "very_high")

    fig, ax = plt.subplots(figsize=FIGSIZE)

    pt_agents = df[df["used_public_transport"]].copy()

    if len(pt_agents) > 0 and "vulnerability_level" in pt_agents.columns:
        pt_agents["vulnerability_level"] = pd.Categorical(
            pt_agents["vulnerability_level"], categories=vuln_order, ordered=True
        )

        counts = (
            pt_agents["vulnerability_level"]
            .value_counts()
            .reindex(vuln_order)
            .fillna(0)
        )

        # Horizontal bars for readability
        bars = ax.barh(
            counts.index.astype(str),
            counts.values,
            color=primary_color,
            edgecolor="k",
            linewidth=0.6,
        )
        ax.set_title(
            "Public Transport Users by Vulnerability Level", fontsize=TITLE_FONTSIZE
        )
        ax.set_xlabel("Number of agents")
        ax.set_ylabel("Vulnerability level")
        setup_academic_grid(ax, axis="x")

        # Annotate counts
        max_val = counts.values.max() if len(counts.values) > 0 else 0
        for i, (label, val) in enumerate(counts.items()):
            ax.text(
                val + max(1, max_val * 0.01),
                i,
                f"{int(val):,}",
                va="center",
                fontsize=BASE_FONTSIZE - 2,
            )
    else:
        ax.text(
            0.5,
            0.5,
            "No public-transport users or missing vulnerability info",
            ha="center",
            va="center",
            fontsize=BASE_FONTSIZE,
        )

    plt.tight_layout()

    if save_path:
        savefig_standard(save_path)

    return fig


# %%
# NEW VISUALIZATIONS FOR DEMOGRAPHIC VARIABLES


def plot_success_by_age(
    agent_summary: pl.DataFrame, save_path: Optional[str] = None
) -> plt.Figure:
    """Plot success rate by age groups"""
    df = agent_summary.to_pandas()

    fig, ax = plt.subplots(figsize=FIGSIZE)

    if "AGE" in df.columns and "success" in df.columns:
        # Create age groups
        df["age_group"] = pd.cut(
            df["AGE"],
            bins=[0, 18, 25, 35, 45, 55, 65, 75, 100],
            labels=[
                "0-17",
                "18-24",
                "25-34",
                "35-44",
                "45-54",
                "55-64",
                "65-74",
                "75+",
            ],
        )

        # Calculate success rate by age group
        success_by_age = df.groupby("age_group")["success"].agg(["mean", "count"])
        success_by_age["mean_pct"] = success_by_age["mean"] * 100

        # Plot
        bars = ax.bar(
            success_by_age.index.astype(str),
            success_by_age["mean_pct"],
            color=primary_color,
            edgecolor="k",
            linewidth=0.6,
            alpha=0.9,
        )

        ax.set_title("Success Rate by Age Group", fontsize=TITLE_FONTSIZE)
        ax.set_ylabel("Success Rate (%)")
        ax.set_xlabel("Age Group")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100.0))
        ax.set_ylim(0, 100)
        ax.tick_params(axis="x", rotation=45)
        setup_academic_grid(ax, axis="y")

        # Annotate bars
        for i, (age_group, row) in enumerate(success_by_age.iterrows()):
            pct = row["mean_pct"]
            n = int(row["count"])
            ax.text(
                i,
                pct + (ax.get_ylim()[1] * 0.02),
                f"{pct:.1f}%\n(n={n})",
                ha="center",
                va="bottom",
                fontsize=BASE_FONTSIZE - 3,
            )
    else:
        ax.text(
            0.5,
            0.5,
            "Missing 'AGE' or 'success' data",
            ha="center",
            va="center",
            fontsize=BASE_FONTSIZE,
        )

    plt.tight_layout()

    if save_path:
        savefig_standard(save_path)

    return fig


def plot_success_by_sex(
    agent_summary: pl.DataFrame, save_path: Optional[str] = None
) -> plt.Figure:
    """Plot success rate by sex"""
    df = agent_summary.to_pandas()

    fig, ax = plt.subplots(figsize=FIGSIZE)

    if "SEX" in df.columns and "success" in df.columns:
        # Map sex codes to labels
        sex_mapping = {1: "Male", 2: "Female"}
        df["sex_label"] = df["SEX"].map(sex_mapping)

        # Calculate success rate by sex
        success_by_sex = df.groupby("sex_label")["success"].agg(["mean", "count"])
        success_by_sex["mean_pct"] = success_by_sex["mean"] * 100

        # Plot
        bars = ax.bar(
            success_by_sex.index.astype(str),
            success_by_sex["mean_pct"],
            color=[primary_color, secondary_color],
            edgecolor="k",
            linewidth=0.6,
            alpha=0.9,
        )

        ax.set_title("Success Rate by Sex", fontsize=TITLE_FONTSIZE)
        ax.set_ylabel("Success Rate (%)")
        ax.set_xlabel("Sex")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100.0))
        ax.set_ylim(0, 100)
        setup_academic_grid(ax, axis="y")

        # Annotate bars
        for i, (sex, row) in enumerate(success_by_sex.iterrows()):
            pct = row["mean_pct"]
            n = int(row["count"])
            ax.text(
                i,
                pct + (ax.get_ylim()[1] * 0.02),
                f"{pct:.1f}%\n(n={n})",
                ha="center",
                va="bottom",
                fontsize=BASE_FONTSIZE - 3,
            )
    else:
        ax.text(
            0.5,
            0.5,
            "Missing 'SEX' or 'success' data",
            ha="center",
            va="center",
            fontsize=BASE_FONTSIZE,
        )

    plt.tight_layout()

    if save_path:
        savefig_standard(save_path)

    return fig


def plot_evacuation_time_by_age(
    agent_summary: pl.DataFrame, save_path: Optional[str] = None
) -> plt.Figure:
    """Plot evacuation time by age groups for successful agents"""
    df = agent_summary.to_pandas()

    fig, ax = plt.subplots(figsize=FIGSIZE)

    successful = df[df["success"]].copy()

    if (
        "AGE" in successful.columns
        and "evacuation_time_minutes" in successful.columns
        and len(successful) > 0
    ):
        # Create age groups
        successful["age_group"] = pd.cut(
            successful["AGE"],
            bins=[0, 18, 25, 35, 45, 55, 65, 75, 100],
            labels=[
                "0-17",
                "18-24",
                "25-34",
                "35-44",
                "45-54",
                "55-64",
                "65-74",
                "75+",
            ],
        )

        # Order by median evacuation time
        order = (
            successful.groupby("age_group")["evacuation_time_minutes"]
            .median()
            .sort_values()
            .index.tolist()
        )

        # Plot
        sns.boxplot(
            data=successful,
            x="age_group",
            y="evacuation_time_minutes",
            order=order,
            ax=ax,
            showfliers=True,
            linewidth=1.2,
            palette="pastel",
        )

        ax.set_title(
            "Evacuation Time by Age Group (Successful Agents)", fontsize=TITLE_FONTSIZE
        )
        ax.set_xlabel("Age Group")
        ax.set_ylabel("Evacuation time (minutes)")
        ax.tick_params(axis="x", rotation=45)
        setup_academic_grid(ax, axis="y")

        # Compute group means and overlay
        means_series = (
            successful.groupby("age_group")["evacuation_time_minutes"]
            .mean()
            .reindex(order)
        )
        means = means_series.values
        x_positions = np.arange(len(order))

        ax.scatter(
            x_positions,
            means,
            marker="D",
            s=MARKER_SIZE,
            edgecolor=primary_color,
            facecolor="white",
            linewidth=1.5,
            zorder=10,
        )

        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        for i, m in enumerate(means):
            ax.text(
                x_positions[i],
                m + y_range * 0.03,
                f"{m:.1f} min",
                ha="center",
                va="bottom",
                fontsize=BASE_FONTSIZE - 3,
                fontweight="bold",
                color=primary_color,
            )
    else:
        ax.text(
            0.5,
            0.5,
            "Missing 'AGE' or 'evacuation_time_minutes' data",
            ha="center",
            va="center",
            fontsize=BASE_FONTSIZE,
        )

    plt.tight_layout()

    if save_path:
        savefig_standard(save_path)

    return fig


def plot_mode_by_age(
    agent_summary: pl.DataFrame, save_path: Optional[str] = None
) -> plt.Figure:
    """Plot transport mode distribution by age groups"""
    df = agent_summary.to_pandas()

    fig, ax = plt.subplots(figsize=FIGSIZE)

    if "AGE" in df.columns and "final_mode" in df.columns:
        # Create age groups
        df["age_group"] = pd.cut(
            df["AGE"],
            bins=[0, 18, 25, 35, 45, 55, 65, 75, 100],
            labels=[
                "0-17",
                "18-24",
                "25-34",
                "35-44",
                "45-54",
                "55-64",
                "65-74",
                "75+",
            ],
        )

        # Count modes by age group
        count_df = (
            df.groupby(["age_group", "final_mode"])
            .size()
            .reset_index(name="count")
            .pivot(index="age_group", columns="final_mode", values="count")
            .fillna(0)
        )

        long = count_df.reset_index().melt(
            id_vars="age_group", var_name="final_mode", value_name="count"
        )

        total_per_mode = long.groupby("final_mode")["count"].sum()
        modes = total_per_mode[total_per_mode > 0].index.tolist()
        palette = sns.color_palette("muted", len(modes))

        # Plot
        sns.barplot(
            data=long[long["final_mode"].isin(modes)],
            x="age_group",
            y="count",
            hue="final_mode",
            ax=ax,
            palette=palette,
            edgecolor="k",
            linewidth=0.6,
        )

        ax.set_title("Transport Mode by Age Group", fontsize=TITLE_FONTSIZE)
        ax.set_ylabel("Count")
        ax.set_xlabel("Age Group")
        ax.legend(title="Transport mode", bbox_to_anchor=(1.02, 1), loc="upper left")
        ax.tick_params(axis="x", rotation=45)
        setup_academic_grid(ax, axis="y")

        # Annotate counts
        for p in ax.patches:
            h = p.get_height()
            if h > 0:
                ax.text(
                    p.get_x() + p.get_width() / 2,
                    h + max(1, h * 0.01),
                    f"{int(h):,}",
                    ha="center",
                    va="bottom",
                    fontsize=BASE_FONTSIZE - 3,
                )
    else:
        ax.text(
            0.5,
            0.5,
            "Missing 'AGE' or 'final_mode' data",
            ha="center",
            va="center",
            fontsize=BASE_FONTSIZE,
        )

    plt.tight_layout()

    if save_path:
        savefig_standard(save_path)

    return fig


# %%
# MAIN FUNCTION TO GENERATE ALL VISUALIZATIONS


def generate_all_visualizations(
    agent_summary: pl.DataFrame,
    journey_segments: pl.DataFrame,
    base_dir: str = BASE_SAVE_DIR,
):
    """Generate and save all individual visualizations"""

    # Create directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)

    # Original visualizations
    print("Generating mode distribution plot...")
    fig = plot_mode_distribution(
        agent_summary, os.path.join(base_dir, "mode_distribution.png")
    )
    plt.close(fig)

    print("Generating success rate by mode plot...")
    fig = plot_success_rate_by_mode(
        agent_summary, os.path.join(base_dir, "success_rate_by_mode.png")
    )
    plt.close(fig)

    print("Generating evacuation time by mode plot...")
    fig = plot_evacuation_time_by_mode(
        agent_summary, os.path.join(base_dir, "evacuation_time_by_mode.png")
    )
    plt.close(fig)

    print("Generating destination types plot...")
    fig = plot_destination_types(
        agent_summary, os.path.join(base_dir, "destination_types.png")
    )
    plt.close(fig)

    print("Generating mode by vulnerability plot...")
    fig = plot_mode_by_vulnerability(
        agent_summary, os.path.join(base_dir, "mode_by_vulnerability.png")
    )
    plt.close(fig)

    print("Generating success rate by vulnerability plot...")
    fig = plot_success_rate_by_vulnerability(
        agent_summary, os.path.join(base_dir, "success_rate_by_vulnerability.png")
    )
    plt.close(fig)

    print("Generating evacuation time by vulnerability plot...")
    fig = plot_evacuation_time_by_vulnerability(
        agent_summary, os.path.join(base_dir, "evacuation_time_by_vulnerability.png")
    )
    plt.close(fig)

    print("Generating distance by vulnerability plot...")
    fig = plot_distance_by_vulnerability(
        agent_summary, os.path.join(base_dir, "distance_by_vulnerability.png")
    )
    plt.close(fig)

    print("Generating PT users by vulnerability plot...")
    fig = plot_pt_users_by_vulnerability(
        agent_summary, os.path.join(base_dir, "pt_users_by_vulnerability.png")
    )
    plt.close(fig)

    # New demographic visualizations
    print("Generating success by age plot...")
    fig = plot_success_by_age(
        agent_summary, os.path.join(base_dir, "success_by_age.png")
    )
    plt.close(fig)

    print("Generating success by sex plot...")
    fig = plot_success_by_sex(
        agent_summary, os.path.join(base_dir, "success_by_sex.png")
    )
    plt.close(fig)

    print("Generating evacuation time by age plot...")
    fig = plot_evacuation_time_by_age(
        agent_summary, os.path.join(base_dir, "evacuation_time_by_age.png")
    )
    plt.close(fig)

    print("Generating mode by age plot...")
    fig = plot_mode_by_age(agent_summary, os.path.join(base_dir, "mode_by_age.png"))
    plt.close(fig)

    print("All visualizations saved to:", base_dir)


# %%
# RUN THE VISUALIZATION GENERATION
generate_all_visualizations(agent_summary, journey_segments)
