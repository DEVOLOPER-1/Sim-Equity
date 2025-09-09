"""
Enhanced academic visualization with SVI-based coloring using coolwarm palette
"""

import math
import os
from collections import defaultdict

import folium
import geojson
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import polars as pl
from playwright.sync_api import sync_playwright
from shapely.geometry import mapping
from sklearn.cluster import DBSCAN

# Mode-specific styling
MODE_STYLES = {
    "VEHICLE": {"weight": 3, "opacity": 0.7, "dashArray": None, "color": "#2E86AB"},
    "BIKE": {"weight": 2, "opacity": 0.7, "dashArray": None, "color": "#43AA8B"},
    "WALKING": {"weight": 2, "opacity": 0.7, "dashArray": None, "color": "#B95F89"},
    "PUBLIC_TRANSPORT": {
        "weight": 2,
        "opacity": 0.7,
        "dashArray": "5, 10",
        "color": "#F39C12",
    },
}


def load_agent_traces_with_svi():
    """
    Load agent traces with SVI information and additional metadata

    Returns:
        dict: Dictionary with agent IDs as keys and their full trace data as values
    """
    print("Loading agent traces with SVI...")

    # Read agent statistics to get SVI values
    try:
        stats_df = pl.read_csv("simulation_outcomes/Agents_Statistics_Trial.csv")
        svi_mapping = {
            row["agent_id"]: row.get("svi", 0.5)
            for row in stats_df.iter_rows(named=True)
        }
        print(f"Loaded SVI for {len(svi_mapping)} agents")
    except Exception as e:
        print(f"Error reading agent statistics: {e}")
        svi_mapping = {}

    agent_traces = {}
    traces_loaded = 0
    all_entries = os.listdir("simulation_outcomes/agents_traces")

    for ag in all_entries:
        try:
            trace_file = f"simulation_outcomes/agents_traces/{ag}"
            if not os.path.exists(trace_file):
                continue

            df = pl.read_csv(trace_file)
            if df.is_empty():
                continue

            # Extract agent ID from filename
            agent_id = ag.replace(".csv", "").replace("agent_", "")
            svi_agent_id = "_".join(ag.replace(".csv", "").split("_")[:-1])

            # Get SVI for this agent
            svi = svi_mapping.get(svi_agent_id, 0.5)

            # Determine coordinate columns (similar to previous approach)
            if "x" not in df.columns or "y" not in df.columns:
                continue

            # Sample coordinates to determine lat/lon
            sample_x = None
            sample_y = None
            try:
                for r in df.select(["x", "y"]).iter_rows():
                    if r[0] is not None and r[1] is not None:
                        sample_x = float(r[0])
                        sample_y = float(r[1])
                        break
            except Exception:
                continue

            if sample_x is None or sample_y is None:
                continue

            # Determine which is lat and which is lon
            if 40 <= sample_x <= 55 and -20 <= sample_y <= 20:
                lat_col, lon_col = "x", "y"
            elif 40 <= sample_y <= 55 and -20 <= sample_x <= 20:
                lat_col, lon_col = "y", "x"
            else:
                lat_col, lon_col = "y", "x"

            # Extract all data
            coordinates = []
            modes = []
            public_transport = []
            timestamps = []

            for row in df.iter_rows(named=True):
                try:
                    lat = float(row[lat_col])
                    lon = float(row[lon_col])
                    if abs(lat) < 1e6 and abs(lon) < 1e6:
                        coordinates.append([lat, lon])
                        mode_val = row.get("mode", "UNKNOWN")
                        # Rename CAR to VEHICLE
                        if mode_val == "CAR":
                            mode_val = "VEHICLE"
                        modes.append(mode_val)
                        public_transport.append(
                            row.get("using_public_transport", False)
                        )
                        timestamps.append(row.get("time", 0))
                except (ValueError, TypeError):
                    continue

            if len(coordinates) >= 2:
                agent_traces[agent_id] = {
                    "coordinates": coordinates,
                    "modes": modes,
                    "public_transport": public_transport,
                    "timestamps": timestamps,
                    "svi": svi,
                }
                traces_loaded += 1

        except Exception as e:
            print(f"Error loading trace for agent {ag}: {e}")

    print(f"Successfully loaded traces for {traces_loaded} agents")
    return agent_traces


def get_svi_colormap(agent_traces):
    """Create a colormap based on the min and max SVI values across all agents"""
    svi_values = [data["svi"] for data in agent_traces.values()]
    min_svi = min(svi_values) + 0.2
    max_svi = max(svi_values)

    # Create a coolwarm colormap normalized to the SVI range
    norm = mcolors.Normalize(vmin=min_svi, vmax=max_svi)
    colormap = cm.ScalarMappable(norm=norm, cmap=cm.coolwarm)

    print(f"SVI range: {min_svi:.3f} to {max_svi:.3f}")
    return colormap, min_svi, max_svi


def detect_bottlenecks(agent_traces, mode=None, epsilon=0.01, min_samples=3):
    """
    Detect bottlenecks where multiple agents paths converge

    Args:
        agent_traces: Dictionary of agent traces
        mode: Filter by specific mode (optional)
        epsilon: DBSCAN parameter for spatial clustering
        min_samples: DBSCAN parameter for minimum points in cluster

    Returns:
        list: Bottleneck points with agent counts
    """
    print(f"Detecting bottlenecks for mode: {mode}")

    # Collect all points from agent paths
    all_points = []
    point_agents = []  # Track which agent each point belongs to

    for agent_id, data in agent_traces.items():
        # Filter by mode if specified
        if mode and not any(m == mode for m in data["modes"]):
            continue

        # Add all points from this agent
        for i, coord in enumerate(data["coordinates"]):
            # For efficiency, sample points (every 5th point)
            if i % 5 == 0:
                all_points.append(coord)
                point_agents.append(agent_id)

    if not all_points:
        return []

    # Convert to numpy array for clustering
    points_array = np.array(all_points)

    # Use DBSCAN to find dense clusters
    clustering = DBSCAN(eps=epsilon, min_samples=min_samples).fit(points_array)
    labels = clustering.labels_

    # Count agents in each cluster (excluding noise points with label=-1)
    cluster_agents = defaultdict(set)
    for i, label in enumerate(labels):
        if label != -1:  # Not noise
            cluster_agents[label].add(point_agents[i])

    # Calculate cluster centers and counts
    bottlenecks = []
    for label, agents in cluster_agents.items():
        if len(agents) >= min_samples:  # Only include significant bottlenecks
            cluster_points = points_array[labels == label]
            center = np.mean(cluster_points, axis=0)
            bottlenecks.append(
                {
                    "location": [center[0], center[1]],
                    "agent_count": len(agents),
                    "cluster_points": cluster_points,
                }
            )

    # Sort by agent count
    bottlenecks.sort(key=lambda x: x["agent_count"], reverse=True)

    print(f"Found {len(bottlenecks)} bottlenecks")
    return bottlenecks


def categorize_bottlenecks(bottlenecks):
    """Categorize bottlenecks into three levels based on agent count"""
    if not bottlenecks:
        return []

    counts = [b["agent_count"] for b in bottlenecks]
    min_count, max_count = min(counts), max(counts)

    # Define thresholds for three levels
    if max_count - min_count < 3:
        # Small range, use equal intervals
        threshold1 = min_count + (max_count - min_count) / 3
        threshold2 = min_count + 2 * (max_count - min_count) / 3
    else:
        # Use percentiles
        threshold1 = np.percentile(counts, 33)
        threshold2 = np.percentile(counts, 66)

    categorized = []
    for bottleneck in bottlenecks:
        count = bottleneck["agent_count"]
        if count <= threshold1:
            level = "low"
        elif count <= threshold2:
            level = "medium"
        else:
            level = "high"

        categorized.append(
            {"location": bottleneck["location"], "agent_count": count, "level": level}
        )

    return categorized


def create_mode_specific_map(
    agent_traces,
    mode,
    colormap,
    min_svi,
    max_svi,
    evacuation_area=None,
    geojson_path=None,
    output_html=None,
    output_png=None,
    title_suffix="",
):
    """
    Create a map for a specific transportation mode

    Args:
        agent_traces: Dictionary of agent traces
        mode: Transportation mode to visualize
        colormap: Matplotlib colormap for SVI values
        min_svi: Minimum SVI value across all agents
        max_svi: Maximum SVI value across all agents
        evacuation_area: Shapely polygon of evacuation area
        geojson_path: Path to region GeoJSON file
        output_html: Output HTML filename
        output_png: Output PNG filename
        title_suffix: Additional text for map title
    """
    print(f"Creating map for mode: {mode}")

    # Academic color scheme
    colors = {
        "background": "#FFFFFF",
        "border": "#34495E",
        "evacuation_border": "#E74C3C",
        "text": "#2C3E50",
    }

    # Create map with academic styling
    map_ = folium.Map(
        location=[48.8566, 2.3522],  # Paris coordinates
        zoom_start=9,
        tiles="CartoDB positron",  # Clean, academic base map
        prefer_canvas=True,
        control_scale=True,
        attr="Academic Visualization",
    )

    # Add region boundaries if GeoJSON provided
    if geojson_path and os.path.exists(geojson_path):
        try:
            with open(geojson_path, "r") as file:
                geojson_data = geojson.load(file)

            folium.GeoJson(
                geojson_data,
                name="Region Boundaries",
                style_function=lambda feature: {
                    "fillColor": "transparent",
                    "color": colors["border"],
                    "weight": 1.5,
                    "fillOpacity": 0,
                    "opacity": 0.7,
                },
                tooltip="Île-de-France Region",
            ).add_to(map_)
        except Exception as e:
            print(f"Error loading GeoJSON: {e}")

    # Add evacuation area with dashed border
    if evacuation_area is not None:
        try:
            evacuation_geojson = mapping(evacuation_area)
            folium.GeoJson(
                evacuation_geojson,
                name="Evacuation Area",
                style_function=lambda x: {
                    "fillColor": "transparent",
                    "color": colors["evacuation_border"],
                    "weight": 3,
                    "fillOpacity": 0,
                    "opacity": 0.8,
                    "dashArray": "10, 10",
                },
                tooltip="Evacuation Area",
            ).add_to(map_)
        except Exception as e:
            print(f"Error adding evacuation area: {e}")

    # Filter agents by mode and add their paths
    agents_added = 0
    for agent_id, data in agent_traces.items():
        # Check if agent used this mode at any point
        if mode not in data["modes"]:
            continue

        coordinates = data["coordinates"]
        modes = data["modes"]
        public_transport = data["public_transport"]
        svi = data["svi"]

        # Get color based on SVI value
        rgba_color = colormap.to_rgba(svi)
        color = mcolors.to_hex(rgba_color)

        # Split path into segments based on mode and public transport
        segments = []
        current_segment = []
        current_style = MODE_STYLES[mode].copy()

        for i, (coord, agent_mode, is_public) in enumerate(
            zip(coordinates, modes, public_transport)
        ):
            # Check if we're still in the same mode
            if agent_mode == mode:
                # For WALKING and BIKE, check public transport usage
                if mode in ["WALKING", "BIKE"]:
                    if is_public:
                        segment_style = MODE_STYLES["PUBLIC_TRANSPORT"].copy()
                    else:
                        segment_style = MODE_STYLES[mode].copy()
                else:
                    segment_style = MODE_STYLES[mode].copy()

                # If style changed, finalize current segment and start new one
                if current_segment and segment_style != current_style:
                    segments.append((current_segment, current_style))
                    current_segment = []

                current_style = segment_style
                current_segment.append(coord)
            else:
                # Mode changed, finalize current segment
                if current_segment:
                    segments.append((current_segment, current_style))
                    current_segment = []

        # Add the last segment if it exists
        if current_segment:
            segments.append((current_segment, current_style))

        # Add all segments to map
        for segment, style in segments:
            if len(segment) < 2:
                continue

            # Add SVI-based color to style
            style["color"] = color

            folium.PolyLine(
                segment,
                **style,
                tooltip=f"Agent {agent_id} (SVI: {svi:.3f})",
            ).add_to(map_)

        # Add end marker only (small and subtle)
        if coordinates:
            folium.CircleMarker(
                coordinates[-1],
                radius=3,
                color=color,
                fill=True,
                fillColor=color,
                fill_opacity=0.7,
                tooltip=f"Agent {agent_id} endpoint (SVI: {svi:.3f})",
            ).add_to(map_)

        agents_added += 1

    print(f"Added {agents_added} agents for mode {mode}")

    # Detect and add bottlenecks
    bottlenecks = detect_bottlenecks(agent_traces, mode=mode)
    categorized_bottlenecks = categorize_bottlenecks(bottlenecks)

    # Add bottlenecks to map with adaptive spacing
    added_bottlenecks = set()
    min_distance = 0.02  # Minimum distance between bottleneck markers (degrees)

    for bottleneck in categorized_bottlenecks:
        location = bottleneck["location"]
        count = bottleneck["agent_count"]
        level = bottleneck["level"]

        # Check if too close to existing bottleneck
        too_close = False
        for added_loc in added_bottlenecks:
            dist = math.sqrt(
                (location[0] - added_loc[0]) ** 2 + (location[1] - added_loc[1]) ** 2
            )
            if dist < min_distance:
                too_close = True
                break

        if not too_close:
            # Determine marker size based on level
            if level == "low":
                radius = 8
                color = "#27AE60"
            elif level == "medium":
                radius = 12
                color = "#F39C12"
            else:  # high
                radius = 16
                color = "#E74C3C"

            # Create bottleneck marker
            folium.CircleMarker(
                location,
                radius=radius,
                color=color,
                fill=True,
                fillColor=color,
                fill_opacity=0.7,
                popup=folium.Popup(f"Bottleneck: {count} agents", max_width=200),
                tooltip=f"Bottleneck: {count} agents",
            ).add_to(map_)

            # Add text with agent count
            folium.Marker(
                location,
                icon=folium.DivIcon(
                    html=f'<div style="font-size: 10px; color: white; text-align: center; font-weight: bold;">{count}</div>'
                ),
            ).add_to(map_)

            added_bottlenecks.add(tuple(location))

    print(f"Added {len(added_bottlenecks)} bottleneck markers")

    # Create colorbar for SVI
    svi_low_color = mcolors.to_hex(colormap.to_rgba(min_svi))
    svi_mid_color = mcolors.to_hex(colormap.to_rgba((min_svi + max_svi) / 2))
    svi_high_color = mcolors.to_hex(colormap.to_rgba(max_svi))
    mode_color = MODE_STYLES.get(mode, {}).get("color", "#000000")
    pt_color = MODE_STYLES["PUBLIC_TRANSPORT"]["color"]

    colorbar_html = f"""
    <div style="position: fixed; 
                bottom: 50px; right: 50px; 
                background-color: white; 
                border: 2px solid grey;
                border-radius: 5px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.2);
                z-index: 9999; 
                padding: 15px;
                font-family: Arial;
                font-size: 12px;">
        <h4 style="margin-top: 0; margin-bottom: 10px;">Legend</h4>
        <p style="margin: 2px 0;"><strong>SVI Score</strong></p>
        <div style="width: 100%; height: 20px; background: linear-gradient(to right, 
            {svi_low_color}, 
            {svi_mid_color}, 
            {svi_high_color}); 
            margin-bottom: 5px;"></div>
        <div style="display: flex; justify-content: space-between;">
            <span>{min_svi:.2f}</span>
            <span>{(min_svi + max_svi)/2:.2f}</span>
            <span>{max_svi:.2f}</span>
        </div>
        <p style="margin: 10px 0 5px 0;"><strong>Line Style:</strong></p>
        <p style="margin: 2px 0;"><span style="border-top: 2px solid {mode_color}; display: inline-block; width: 20px;"></span> {mode.title() if mode != 'VEHICLE' else 'Vehicle'}</p>
        <p style="margin: 2px 0;"><span style="border-top: 2px dashed {pt_color}; display: inline-block; width: 20px;"></span> Public Transport</p>
        <p style="margin: 10px 0 5px 0;"><strong>Bottlenecks:</strong></p>
        <p style="margin: 2px 0;"><span style="color: #27AE60;">●</span> Low congestion</p>
        <p style="margin: 2px 0;"><span style="color: #F39C12;">●</span> Medium congestion</p>
        <p style="margin: 2px 0;"><span style="color: #E74C3C;">●</span> High congestion</p>
    </div>
    """

    map_.get_root().html.add_child(folium.Element(colorbar_html))

    # Add title
    mode_display_name = "Vehicle" if mode == "VEHICLE" else mode.title()
    title_html = f"""
    <div style="position: fixed; 
                top: 10px; left: 50px; 
                background-color: white; 
                border: 2px solid grey;
                border-radius: 5px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.2);
                z-index: 9999; 
                padding: 10px 15px;">
        <h4 style="margin: 0; font-family: Arial; font-size: 16px; color: #2C3E50;">
            {mode_display_name} Evacuation Paths{title_suffix}
        </h4>
    </div>
    """
    map_.get_root().html.add_child(folium.Element(title_html))

    # Save outputs
    if output_html:
        try:
            map_.save(output_html)
            print(f"Map saved as HTML: {output_html}")
        except Exception as e:
            print(f"Error saving HTML: {e}")

    if output_png:
        try:
            save_map_as_png(output_html, output_png)
            print(f"Map saved as PNG: {output_png}")
        except Exception as e:
            print(f"Error saving PNG: {e}")

    return map_


def save_map_as_png(html_file, png_file, width=600, height=925, wait_time=5):
    """Convert HTML map to PNG using Playwright"""
    html_path = os.path.abspath(html_file)
    file_url = f"file://{html_path}"

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                context = browser.new_context(
                    viewport={"width": width, "height": height}, device_scale_factor=1
                )
                page = context.new_page()
                page.goto(file_url)
                page.wait_for_timeout(wait_time * 1000)
                page.screenshot(path=png_file, full_page=False, type="png")
            finally:
                browser.close()
    except Exception as e:
        print(f"Error during PNG conversion: {e}")


def main():
    """Main function to create all three mode-specific maps"""
    print("Starting enhanced academic visualization...")

    # Load agent traces with SVI
    agent_traces = load_agent_traces_with_svi()

    # Create colormap based on SVI values
    colormap, min_svi, max_svi = get_svi_colormap(agent_traces)
    # min_svi += 0.2
    # Load evacuation area
    evacuation_polygon = None
    try:
        from simulation.space.evacuation_area_initializer import EnvironmentInitializer

        SCENARIO_CENTER_LAT = 48.858844
        SCENARIO_CENTER_LON = 2.347012
        SCENARIO_RADIUS_KM = 50.0

        env_init = EnvironmentInitializer(
            (SCENARIO_CENTER_LAT, SCENARIO_CENTER_LON), SCENARIO_RADIUS_KM
        )
        evacuation_polygon = env_init.get_made_polygon
        print("Successfully loaded evacuation area")
    except Exception as e:
        print(f"Could not load evacuation area: {e}")

    # Check GeoJSON file
    geojson_path = "data/departements-ile-de-france.geojson"
    if not os.path.exists(geojson_path):
        print(f"Warning: GeoJSON file not found at {geojson_path}")
        geojson_path = None

    # Create maps for each mode
    modes = ["VEHICLE", "BIKE", "WALKING"]

    for mode in modes:
        output_html = f"plots/evacuation_maps/evacuation_map_{mode.lower()}.html"
        output_png = f"plots/evacuation_maps/evacuation_map_{mode.lower()}.png"

        create_mode_specific_map(
            agent_traces=agent_traces,
            mode=mode,
            colormap=colormap,
            min_svi=min_svi,
            max_svi=max_svi,
            evacuation_area=evacuation_polygon,
            geojson_path=geojson_path,
            output_html=output_html,
            output_png=output_png,
            title_suffix=" - Île-de-France Region",
        )

    print("\n" + "=" * 60)
    print("ACADEMIC VISUALIZATION COMPLETE!")
    print("Created three mode-specific maps:")
    for mode in modes:
        print(f"  - evacuation_map_{mode.lower()}.html")
        print(f"  - evacuation_map_{mode.lower()}.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
