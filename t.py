"""
Fixed visualization script with proper coordinate handling and debugging
"""

import colorsys
import os
import random

import folium
import geojson
import polars as pl
from playwright.sync_api import sync_playwright
from shapely.geometry import mapping


def load_agent_traces():
    """
    Load all agent traces from CSV files with improved coordinate handling.

    Returns:
        dict: Dictionary with agent IDs as keys and their path data as values
    """
    print("Loading agent traces...")

    # Read agent IDs from statistics file
    try:
        agents_ids = pl.Series(
            pl.read_csv("simulation_outcomes/Agents_Statistics_Trial.csv").select(
                "agent_id"
            )
        ).to_list()
    except Exception as e:
        print(f"Error reading agent statistics: {e}")
        return {}

    agent_traces = {}
    traces_loaded = 0
    coordinate_issues = 0

    for ag in agents_ids:
        try:
            trace_file = f"simulation_outcomes/agents_traces/{ag}.csv"
            if not os.path.exists(trace_file):
                print(f"Trace file not found for agent {ag}")
                continue

            df = pl.read_csv(trace_file)

            # Check if dataframe is empty
            if df.is_empty():
                print(f"Empty trace file for agent {ag}")
                continue

            # Debug: Print first few rows to understand data structure
            if traces_loaded == 0:  # Only debug first agent
                print(f"\nDEBUG - First agent ({ag}) data structure:")
                print(f"Columns: {df.columns}")
                print(f"Shape: {df.shape}")
                print("First 3 rows:")
                print(df.head(3))

            # Extract coordinates - the CSV has 'y' (latitude) and 'x' (longitude)
            # We need to convert to [lat, lon] format for Folium
            coordinates_df = df.select(["y", "x"]).drop_nulls()

            if coordinates_df.is_empty():
                print(f"No valid coordinates for agent {ag}")
                continue

            # Convert to list of [lat, lon] pairs
            coordinates = []
            for row in coordinates_df.iter_rows():
                lat, lon = float(row[0]), float(row[1])  # y=lat, x=lon

                # Validate coordinate ranges
                if lat and lon:
                    coordinates.append([lat, lon])  # Folium expects [lat, lon]
                else:
                    coordinate_issues += 1
                    if coordinate_issues <= 5:  # Only print first few warnings
                        print(
                            f"Invalid coordinates for agent {ag}: lat={lat}, lon={lon}"
                        )

            if len(coordinates) >= 2:  # Need at least 2 points for a path
                agent_traces[ag] = coordinates
                traces_loaded += 1

                # Debug first agent's coordinates
                if traces_loaded == 1:
                    print(f"DEBUG - Agent {ag} coordinates sample:")
                    print(f"  Total points: {len(coordinates)}")
                    print(f"  First point: {coordinates[0]} (lat, lon)")
                    print(f"  Last point: {coordinates[-1]} (lat, lon)")
                    print(f"  Path length: {len(coordinates)} points")
            else:
                print(
                    f"Agent {ag}: insufficient valid coordinates ({len(coordinates)} points)"
                )

        except Exception as e:
            print(f"Error loading trace for agent {ag}: {e}")

    print(f"\nSUMMARY:")
    print(
        f"Successfully loaded traces for {traces_loaded} out of {len(agents_ids)} agents"
    )
    print(f"Coordinate issues found: {coordinate_issues}")
    return agent_traces


def debug_evacuation_area(evacuation_area):
    """Debug function to check evacuation area properties"""
    print("\n=== DEBUGGING EVACUATION AREA ===")

    if evacuation_area is None:
        print("ERROR: evacuation_area is None")
        return False

    print(f"Type: {type(evacuation_area)}")

    try:
        from shapely.geometry.base import BaseGeometry

        if not isinstance(evacuation_area, BaseGeometry):
            print(f"ERROR: Not a Shapely geometry object")
            return False

        print(f"Geometry type: {evacuation_area.geom_type}")
        print(f"Is valid: {evacuation_area.is_valid}")
        print(f"Is empty: {evacuation_area.is_empty}")

        # Get bounds
        bounds = evacuation_area.bounds
        print(f"Bounds (minx, miny, maxx, maxy): {bounds}")

        # Convert to GeoJSON to check structure
        geojson_dict = mapping(evacuation_area)
        print(f"GeoJSON type: {geojson_dict.get('type')}")

        if geojson_dict.get("type") == "Polygon":
            coords = geojson_dict.get("coordinates", [])
            if coords:
                exterior_ring = coords[0]
                print(f"Number of exterior points: {len(exterior_ring)}")
                print(f"First few points: {exterior_ring[:3]}")

                # Check coordinate order and values
                first_point = exterior_ring[0]
                if len(first_point) >= 2:
                    x, y = first_point[0], first_point[1]  # GeoJSON is [lon, lat]
                    print(f"First point (GeoJSON format) - lon: {x}, lat: {y}")

                    # Validate that coordinates are reasonable for Paris area
                    if 1.0 <= x <= 4.0 and 48.0 <= y <= 50.0:
                        print("✅ Coordinates appear correct for Paris region")
                    else:
                        print(f"⚠️ Coordinates may be outside expected Paris region")

        return True

    except Exception as e:
        print(f"ERROR during evacuation area debug: {e}")
        import traceback

        traceback.print_exc()
        return False


def generate_distinct_colors(n):
    """Generate n visually distinct colors"""
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.7 + random.random() * 0.3
        value = 0.7 + random.random() * 0.3
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        color = "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
        colors.append(color)
    return colors


def create_research_map(
    geojson_path: str,
    agent_paths: dict = None,
    evacuation_area=None,
    max_agents_to_display: int = 20,
    output_html: str = "research_map.html",
    output_png: str = "research_map.png",
    title: str = "Île-de-France Region",
):
    """
    Create a clean, research-paper styled map with PNG export capability.
    """
    colors = {
        "primary": "#4C72B0",
        "secondary": "#55A868",
        "highlight": "#C44E52",
        "border": "#8C8C8C",
        "background": "#FFFFFF",
        "evacuation": "#FF6B6B",  # More visible red
        "evacuation_border": "#DC143C",
    }

    # Debug evacuation area first
    if evacuation_area is not None:
        evacuation_valid = debug_evacuation_area(evacuation_area)
        if not evacuation_valid:
            print("WARNING: Evacuation area has issues, proceeding without it")
            evacuation_area = None

    # Load the GeoJSON data
    try:
        with open(geojson_path, "r") as file:
            geojson_data = geojson.load(file)
        print("✅ Successfully loaded region GeoJSON")
    except Exception as e:
        print(f"ERROR loading GeoJSON: {e}")
        geojson_data = None

    # Create map with clean styling - start with a wider view
    map_ = folium.Map(
        location=[48.8566, 2.3522],  # Paris coordinates
        zoom_start=8,  # Wider initial zoom
        tiles="cartodb positron",
        prefer_canvas=True,
        control_scale=True,
    )

    # Add the GeoJSON region boundaries
    if geojson_data:
        geo_json = folium.GeoJson(
            geojson_data,
            name="Region Boundaries",
            style_function=lambda feature: {
                "fillColor": colors["primary"],
                "color": colors["border"],
                "weight": 2,
                "fillOpacity": 0.1,
                "opacity": 0.8,
            },
            tooltip="Île-de-France Region",
        ).add_to(map_)

        # Fit map bounds to the GeoJSON
        try:
            map_.fit_bounds(geo_json.get_bounds(), padding=(20, 20))
            print("✅ Added region boundaries and fitted bounds")
        except Exception as e:
            print(f"Warning: Could not fit bounds to GeoJSON: {e}")

    # Add evacuation area
    if evacuation_area is not None:
        try:
            print("Adding evacuation area to map...")
            evacuation_geojson = mapping(evacuation_area)

            evacuation_layer = folium.GeoJson(
                evacuation_geojson,
                name="🚨 Evacuation Area",
                style_function=lambda x: {
                    "fillColor": colors["evacuation"],
                    "color": colors["evacuation_border"],
                    "weight": 3,
                    "fillOpacity": 0.3,
                    "opacity": 1.0,
                },
                popup=folium.Popup("EVACUATION AREA", max_width=200),
                tooltip="🚨 EVACUATION AREA 🚨",
            )
            evacuation_layer.add_to(map_)
            print("✅ Successfully added evacuation area to map")

        except Exception as e:
            print(f"ERROR adding evacuation area: {e}")

    # Add agent paths with enhanced debugging
    if agent_paths and len(agent_paths) > 0:
        print(f"\n=== ADDING AGENT PATHS ===")
        print(f"Total agent paths available: {len(agent_paths)}")

        # Create a feature group for agent paths
        agent_layer = folium.FeatureGroup(name="Agent Paths", show=True)

        # Get agent IDs and limit display
        agent_ids = list(agent_paths.keys())[:max_agents_to_display]
        path_colors = generate_distinct_colors(len(agent_ids))

        paths_added = 0
        valid_paths = 0

        for idx, agent_id in enumerate(agent_ids):
            path = agent_paths[agent_id]
            color = path_colors[idx]

            # Debug first few agents
            if idx < 3:
                print(f"\nDEBUG Agent {agent_id}:")
                print(f"  Path length: {len(path)} points")
                print(f"  First point: {path[0]} (lat={path[0][0]}, lon={path[0][1]})")
                print(
                    f"  Last point: {path[-1]} (lat={path[-1][0]}, lon={path[-1][1]})"
                )
                print(f"  Color: {color}")

                # Validate coordinates are reasonable for Paris
                lat_check = 48.0 <= path[0][0] <= 50.0
                lon_check = 1.0 <= path[0][1] <= 4.0
                print(
                    f"  Coordinate validation: lat_ok={lat_check}, lon_ok={lon_check}"
                )

            if len(path) < 2:
                print(f"Skipping agent {agent_id}: insufficient points ({len(path)})")
                continue

            try:
                # Validate that coordinates are reasonable
                first_point = path[0]
                if len(first_point) != 2:
                    print(f"Invalid point format for agent {agent_id}: {first_point}")
                    continue

                lat, lon = first_point[0], first_point[1]
                if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                    print(
                        f"Invalid coordinates for agent {agent_id}: lat={lat}, lon={lon}"
                    )
                    continue

                # Check if path is within reasonable bounds for Paris region
                if not (47.0 <= lat <= 50.0 and 1.0 <= lon <= 4.0):
                    print(
                        f"Agent {agent_id} outside Paris region: lat={lat}, lon={lon}"
                    )
                    # Continue anyway, might be valid

                valid_paths += 1

                # Add path line with higher weight for visibility
                folium.PolyLine(
                    path,
                    color=color,
                    weight=3,  # Increased weight
                    opacity=0.8,  # Higher opacity
                    tooltip=f"Agent {agent_id} (Path)",
                ).add_to(agent_layer)

                # Add start marker (green)
                folium.CircleMarker(
                    path[0],
                    radius=8,  # Larger radius
                    color="green",
                    fill=True,
                    fillColor="lightgreen",
                    fill_opacity=0.8,
                    tooltip=f"Agent {agent_id} START",
                ).add_to(agent_layer)

                # Add end marker (red)
                folium.CircleMarker(
                    path[-1],
                    radius=8,  # Larger radius
                    color="red",
                    fill=True,
                    fillColor="lightcoral",
                    fill_opacity=0.8,
                    tooltip=f"Agent {agent_id} END",
                ).add_to(agent_layer)

                paths_added += 1

            except Exception as e:
                print(f"Error adding path for agent {agent_id}: {e}")

        # Add the layer to the map
        agent_layer.add_to(map_)

        print(f"\n=== AGENT PATHS SUMMARY ===")
        print(f"Valid paths found: {valid_paths}")
        print(f"Paths successfully added: {paths_added}")
        print(f"✅ Agent paths layer added to map")

        # If no paths were added, add a debug marker
        if paths_added == 0:
            print("⚠️ NO PATHS ADDED - Adding debug marker at Paris center")
            folium.Marker(
                [48.8566, 2.3522],
                popup="DEBUG: No agent paths were displayed",
                tooltip="Debug Marker",
                icon=folium.Icon(color="red", icon="exclamation-sign"),
            ).add_to(map_)

    else:
        print("No agent paths provided")

    # Add layer control
    folium.LayerControl().add_to(map_)

    # Add title
    title_html = f"""
        <div style="position: fixed; 
                    top: 10px; left: 50px; 
                    background-color: white; 
                    border: 2px solid #333;
                    border-radius: 5px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.3);
                    z-index: 9999; 
                    padding: 15px;">
            <h4 style="margin: 0; font-family: Arial; font-size: 16px; color: #333; font-weight: bold;">{title}</h4>
        </div>
        """
    map_.get_root().html.add_child(folium.Element(title_html))

    # Save HTML
    try:
        map_.save(output_html)
        print(f"✅ Map saved as HTML: {output_html}")
    except Exception as e:
        print(f"ERROR saving HTML: {e}")

    # Save as PNG
    try:
        save_map_as_png(output_html, output_png)
        print(f"✅ Map saved as PNG: {output_png}")
    except Exception as e:
        print(f"ERROR saving PNG: {e}")

    return map_


def save_map_as_png(
    html_file: str,
    png_file: str,
    width: int = 1200,
    height: int = 900,
    wait_time: int = 5,
):
    """Convert HTML map to PNG using Playwright."""
    html_path = os.path.abspath(html_file)
    file_url = f"file://{html_path}"

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                context = browser.new_context(
                    viewport={"width": width, "height": height}
                )
                page = context.new_page()
                page.goto(file_url)
                page.wait_for_timeout(wait_time * 1000)
                page.screenshot(path=png_file, full_page=True)
            finally:
                browser.close()
    except Exception as e:
        print(f"Error during PNG conversion: {e}")


# Example usage
if __name__ == "__main__":
    print("Starting map visualization with enhanced debugging...")

    # Load agent traces with improved debugging
    agent_traces = load_agent_traces()

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
        print("✅ Successfully loaded actual evacuation area")

    except Exception as e:
        print(f"Could not load actual evacuation area: {e}")

    # Check GeoJSON file
    geojson_path = "data/departements-ile-de-france.geojson"
    if not os.path.exists(geojson_path):
        print(f"WARNING: GeoJSON file not found at {geojson_path}")

    # Create the map with enhanced settings
    try:
        research_map = create_research_map(
            geojson_path=geojson_path,
            agent_paths=agent_traces,
            evacuation_area=evacuation_polygon,
            max_agents_to_display=100,  # Show more agents
            output_html="agent_evacuation_paths_debug.html",
            output_png="agent_evacuation_paths_debug.png",
            title="Île-de-France Region: Agent Evacuation Paths (Debug Version)",
        )

        print("\n" + "=" * 50)
        print("🎉 DEBUG MAP CREATED SUCCESSFULLY!")
        print("=" * 50)
        print("Check the files:")
        print("- agent_evacuation_paths_debug.html")
        print("- agent_evacuation_paths_debug.png")
        print("=" * 50)

    except Exception as e:
        print(f"ERROR creating map: {e}")
        import traceback

        traceback.print_exc()
