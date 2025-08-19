# FILE: evacuation_model.py
# -----------------------------
# This module defines the core agent-based model for the evacuation simulation.
# It has been refactored for performance, clarity, and correctness, using OSMnx
# for efficient spatial operations and a robust state machine for agent behavior.

import gc
import time
import warnings
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import mesa
import networkx as nx
import osmnx as ox
import polars as pl
import tqdm
from rustworkx import PathMapping
from scipy.spatial import KDTree
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.strtree import STRtree

# Configure OSMnx for better performance
ox.settings.use_cache = True
ox.settings.log_console = False
ox.settings.timeout = 300


class HybridGraphManager:
    """
    Uses NetworkX for reliable pathfinding and OSMnx for fast spatial operations
    """

    def __init__(self, graphml_path: str):
        print("🔄 Loading graphs in hybrid mode...")
        start = time.time()

        # Load NetworkX for pathfinding (reliable)
        self.nx_graph = ox.load_graphml(graphml_path)
        print(f"   NetworkX loaded: {self.nx_graph.number_of_nodes()} nodes")

        # Convert numeric attributes to floats
        self._convert_numeric_attributes()

        # Build spatial index from NetworkX data (fast spatial queries)
        self._build_spatial_index()

        # Precompute node coordinates for OSMnx functions
        self._extract_node_coordinates()

        load_time = time.time() - start
        print(f"✅ Hybrid graphs loaded in {load_time:.2f}s")

    def _extract_node_coordinates(self):
        """Extract node coordinates for efficient OSMnx operations"""
        self.node_coords = {}
        self.node_ids = []
        self.node_points = []

        for node_id, data in self.nx_graph.nodes(data=True):
            if "x" in data and "y" in data:
                try:
                    x, y = float(data["x"]), float(data["y"])
                    self.node_coords[node_id] = (y, x)  # (lat, lon)
                    self.node_ids.append(node_id)
                    self.node_points.append((y, x))
                except (ValueError, TypeError):
                    continue

        # Create KDTree for fast spatial queries
        if self.node_points:
            self.kdtree = KDTree(self.node_points)
        else:
            self.kdtree = None

    def _convert_numeric_attributes(self):
        """Convert known numeric attributes to floats"""
        # Convert node attributes
        for node, data in self.nx_graph.nodes(data=True):
            for attr in ["x", "y"]:
                if attr in data and isinstance(data[attr], str):
                    try:
                        data[attr] = float(data[attr])
                    except ValueError:
                        pass

        # Convert edge attributes
        for u, v, data in self.nx_graph.edges(data=True):
            for attr in ["length", "capacity", "weight"]:
                if attr in data and isinstance(data[attr], str):
                    try:
                        data[attr] = float(data[attr])
                    except ValueError:
                        pass

    def _build_spatial_index(self):
        """Build STRtree spatial index from NetworkX node coordinates"""
        print("   Building spatial index from NetworkX data...")

        nodes_with_coords = []
        points = []

        for node_id, data in self.nx_graph.nodes(data=True):
            if "x" in data and "y" in data:
                try:
                    x, y = float(data["x"]), float(data["y"])
                    points.append(Point(x, y))
                    nodes_with_coords.append(node_id)
                except (ValueError, TypeError):
                    continue

        self.spatial_index = STRtree(points)
        self.spatial_node_mapping = {
            i: node_id for i, node_id in enumerate(nodes_with_coords)
        }
        print(f"   Spatial index built with {len(points)} nodes")

    def get_nearest_node(self, pos: tuple) -> Optional[int]:
        """Fast spatial search using OSMnx's optimized methods"""
        if pos is None or self.kdtree is None:
            return None

        try:
            # Use OSMnx's efficient nearest node search
            if hasattr(self, "kdtree") and self.kdtree is not None:
                # Find the nearest node using KDTree
                dist, idx = self.kdtree.query([pos], k=1)
                return self.node_ids[idx[0]]
            else:
                # Fallback to manual search
                return self._manual_nearest_node(pos)
        except Exception as e:
            warnings.warn(f"Error in nearest node search: {e}")
            return self._manual_nearest_node(pos)

    def _manual_nearest_node(self, pos: tuple) -> Optional[int]:
        """Fallback manual search"""
        from haversine import haversine

        target_lat, target_lon = float(pos[0]), float(pos[1])
        min_distance = float("inf")
        nearest_node = None

        for node_id, data in self.nx_graph.nodes(data=True):
            if "x" in data and "y" in data:
                try:
                    node_lon = float(data["x"])
                    node_lat = float(data["y"])

                    distance = haversine(
                        (target_lat, target_lon),
                        (node_lat, node_lon),
                        normalize=True,
                        check=True,
                    )

                    if distance < min_distance:
                        min_distance = distance
                        nearest_node = node_id

                except (ValueError, TypeError):
                    continue

        return nearest_node

    def find_shortest_path(
        self, source: int, target: int, weight="length"
    ) -> List[int]:
        """Reliable pathfinding using NetworkX"""
        try:
            path = nx.shortest_path(
                self.nx_graph, source=source, target=target, weight=weight
            )
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def get_edge_data(self, u: int, v: int) -> Optional[dict]:
        """Get edge attributes from NetworkX"""
        try:
            return self.nx_graph[u][v]
        except KeyError:
            return None

    def node_exists(self, node_id: int) -> bool:
        """Check if node exists in graph"""
        return node_id in self.nx_graph

    def get_node_data(self, node_id: int) -> Optional[dict]:
        """Get node attributes"""
        try:
            return self.nx_graph.nodes[node_id]
        except KeyError:
            return None


# --- AGENT DEFINITION ---
class EvacuationAgent(mesa.Agent):
    """
    Represents a person from the NetMob25 dataset with their own unique
    vulnerability, assets, and behavioral parameters.
    """

    def __init__(self, model: mesa.Model, unique_id: str, **kwargs):
        super().__init__(model)
        self.unique_id: str = unique_id
        self.status = "INACTIVE"  # State machine: INACTIVE -> PLANNING -> EVACUATING -> (ARRIVED | FAILED)
        self.svi = float(kwargs.get("SVI_normalized", 0.0))
        self.main_mode = kwargs.get("main_mode", "WALKING")

        # Location handling
        self.home_location = self._get_location(
            kwargs, "home_location_lat", "home_location_lon"
        )
        self.start_pos = self._get_location(kwargs, "start_lat", "start_lon")
        self.current_pos_node = None

        if self.start_pos:
            # Find the starting node on the graph; fail the agent if none is found.
            self.current_pos_node = self.model.get_nearest_node(
                self.start_pos,
                self.main_mode,
            )
            if self.current_pos_node is None:
                warnings.warn(
                    f"Agent {self.unique_id}: Could not find a valid starting node on the graph."
                )
                self.status = "FAILED"
        else:
            warnings.warn(f"Agent {self.unique_id}: Missing start_lat or start_lon.")
            self.status = "FAILED"

        # Routing state
        self.target_node = None
        self.path: List[int] = []
        self.time_on_current_edge_s = 0.0

        # Time management and SVI-driven behavior
        self._init_behavioral_params(kwargs)

    @staticmethod
    def _get_location(
        data: dict, lat_key: str, lon_key: str
    ) -> Optional[Tuple[float, float]]:
        """Safely extract and validate location coordinates from agent data."""
        lat, lon = data.get(lat_key), data.get(lon_key)
        return (float(lat), float(lon)) if lat is not None and lon is not None else None

    def _init_behavioral_params(self, kwargs: dict):
        """Initialize all agent-specific timing and behavioral parameters based on SVI."""
        # 1. Reaction Delay
        start_time_str = kwargs.get("start_time")
        base_activation_time = (
            datetime.fromisoformat(start_time_str)
            if start_time_str
            else self.model.start_datetime
        )
        start_delay_s = self.svi * self.model.max_svi_start_delay_s
        self.effective_activation_time = base_activation_time + timedelta(
            seconds=start_delay_s
        )
        self.evacuation_time = 0

        # 2. Speed Penalty
        base_speed_m_s = self._get_base_speed(kwargs)
        self.speed_m_s = base_speed_m_s * (
            1.0 - self.svi * self.model.svi_speed_penalty
        )

        # 3. Patience/Rerouting
        self.patience_threshold_s = self.model.base_patience_s * (1.0 - self.svi)
        self.time_stuck_s = 0

    def _get_base_speed(self, kwargs: dict) -> float:
        """Get the appropriate base speed from the agent's data based on their main mode."""
        mode_speeds = {
            "WALKING": kwargs.get("walking_speed_m_s", 1.4),  # Avg. human walk speed
            "BIKE": kwargs.get("cycling_speed_m_s", 4.5),  # Avg. city cycling speed
        }
        return float(
            mode_speeds.get(self.main_mode.upper(), kwargs.get("median_speed_m_s", 8.3))
        )  # Default to car speed

    def step(self):
        """Executes the agent's logic for a single simulation time step."""
        if self.status in ["ARRIVED", "FAILED"]:
            return

        if self.status == "INACTIVE":
            if self.model.sim_time >= self.effective_activation_time:
                self.status = "PLANNING"
            else:
                return

        self.evacuation_time += self.model.step_seconds

        if self.status == "PLANNING":
            self.plan_evacuation_route()

        if self.status == "EVACUATING":
            if self.is_stuck_in_traffic():
                self.time_stuck_s += self.model.step_seconds
                if self.time_stuck_s > self.patience_threshold_s:
                    self.status = "PLANNING"  # Trigger replanning
                    self.time_stuck_s = 0
            else:
                self.time_stuck_s = 0
            self.move()

    def plan_evacuation_route(self):
        """Determines the agent's destination and calculates the initial route."""
        if self.status == "FAILED":
            return

        # Destination Logic: Is the agent's home a safe and known destination?
        is_home_safe = self.home_location and not self.model.is_pos_in_evacuation_area(
            self.home_location
        )
        if is_home_safe:
            self.target_node = self.model.get_nearest_node(
                self.home_location, self.main_mode
            )

        # If home is not an option, find the nearest designated shelter.
        if self.target_node is None:
            self.target_node = self.model.get_nearest_shelter_node(
                self.current_pos_node, self.main_mode
            )

        if self.target_node is None:
            warnings.warn(
                f"Agent {self.unique_id}: Could not determine a valid destination."
            )
            self.status = "FAILED"
            return

        # Calculate the path using the model's pathfinding service.
        self.path: PathMapping = self.model.plan_route(
            self, self.current_pos_node, self.target_node
        )

        if self.path:
            self.status = "EVACUATING"
            self.time_on_current_edge_s = 0.0  # Reset edge timer for the new path
        else:
            warnings.warn(
                f"Agent {self.unique_id}: Failed to find a valid path from {self.current_pos_node} to {self.target_node}."
            )
            self.status = "FAILED"

    def move(self):
        """Moves the agent from one node to the next along its calculated path."""
        if not self.path or len(self.path) < 2:
            self.status = (
                "ARRIVED" if self.current_pos_node == self.target_node else "FAILED"
            )
            return

        u, v = self.path[0], self.path[1]

        # Get edge data using the correct rustworkx method
        graph = self.model.hybrid_managers[self.model._normalize_mode(self.main_mode)]
        edge_data = graph.get_edge_data(u, v)
        if edge_data is None:
            warnings.warn(
                f"Agent {self.unique_id}: Edge ({u}, {v}) not found in graph. Replanning."
            )
            self.status = "PLANNING"
            return

        edge_length_m = edge_data.get(
            "length", 1.0
        )  # Default to 1m if length is missing
        agent_speed = max(self.speed_m_s, 0.1)  # Prevent division by zero
        time_to_traverse_edge_s = edge_length_m / agent_speed

        # Report usage for the entire duration of traversal
        self.model.report_edge_usage(self, (u, v))

        if (
            self.time_on_current_edge_s + self.model.step_seconds
            >= time_to_traverse_edge_s
        ):
            # Agent will complete the edge in this step
            self.current_pos_node = v
            self.path.pop(0)  # Advance to the next node in the path
            self.time_on_current_edge_s = 0.0
        else:
            # Agent continues along the current edge
            self.time_on_current_edge_s += self.model.step_seconds

    def is_stuck_in_traffic(self) -> bool:
        """Checks if the agent's next intended edge is congested."""
        if self.main_mode != "CAR" or not self.path or len(self.path) < 2:
            return False

        next_edge = (self.path[0], self.path[1])
        return self.model.get_edge_congestion(next_edge) > 1.0


# --- MODEL DEFINITION ---
class EvacuationModel(mesa.Model):
    """
    The main model class for the evacuation simulation.
    Manages the environment (graphs), agent scheduling, and data collection.
    """

    def __init__(
        self,
        agents_df: pl.DataFrame,
        graphml_path_drive: str,
        graphml_path_walk: str,
        graphml_path_cycle: str,
        amenities_df: pl.DataFrame,
        evacuation_area_polygon: Polygon,
        start_datetime: datetime,
        step_seconds: int = 60,
        svi_speed_penalty: float = 0.5,
        max_svi_start_delay_s: int = 1800,
        base_patience_s: int = 300,
    ):
        super().__init__()
        print("🚀 Initializing EvacuationModel...")
        init_start = time.time()

        self.start_datetime = start_datetime
        self.sim_time = start_datetime
        self.step_seconds = step_seconds

        # Simulation behavioral parameters
        self.svi_speed_penalty = svi_speed_penalty
        self.max_svi_start_delay_s = max_svi_start_delay_s
        self.base_patience_s = base_patience_s

        # Environment data
        print("📊 Setting up hybrid graph managers...")
        self.hybrid_managers = {
            "CAR": HybridGraphManager(graphml_path_drive),
            "WALKING": HybridGraphManager(graphml_path_walk),
            "BIKE": HybridGraphManager(graphml_path_cycle),
        }
        print("✅ Hybrid graph managers created")

        # Environment data
        self.evac_polygon = evacuation_area_polygon

        # Pre-compute and cache the locations of safe shelters
        print("🏠 Pre-computing shelter nodes...")
        shelter_start = time.time()
        self.shelter_nodes = self._precompute_shelter_nodes(amenities_df)
        shelter_time = time.time() - shelter_start
        print(f"✅ Shelter nodes computed in {shelter_time:.2f}s")

        # Precompute shelter distance matrices for each mode
        print("📏 Precomputing shelter distance matrices...")
        dist_matrix_start = time.time()
        self.shelter_distance_matrices = self._precompute_shelter_distance_matrices()
        dist_matrix_time = time.time() - dist_matrix_start
        print(f"✅ Shelter distance matrices computed in {dist_matrix_time:.2f}s")

        # Agent scheduling
        print("👥 Creating agents...")
        agent_start = time.time()
        agent_count = 0
        failed_agents = 0

        for agent_data in tqdm.tqdm(
            agents_df.iter_rows(named=True),
            total=agents_df.shape[0],
            desc="👥 Creating Agents",
            unit="agent",
            unit_scale=True,
            ncols=120,
            colour="blue",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
            ascii=False,
            dynamic_ncols=True,
            smoothing=0.3,
            mininterval=0.1,
            maxinterval=1.0,
            leave=True,
        ):
            agent_id = agent_data.get("ID")
            if agent_id:
                agent = EvacuationAgent(
                    model=self, unique_id=str(agent_id), **agent_data
                )
                agent_count += 1
                if agent.status == "FAILED":
                    failed_agents += 1

                if agent_count % 10000 == 0:
                    print(f"   Created {agent_count:,} agents...")

        agent_time = time.time() - agent_start
        print(
            f"✅ Created {agent_count:,} agents ({failed_agents} failed) in {agent_time:.2f}s"
        )

        # Bottleneck monitoring and data collection
        print("📈 Setting up data collection...")
        self.edge_load = defaultdict(int)
        self.edge_agents = defaultdict(list)
        self.bottleneck_log: List[Dict[str, Any]] = []

        self.datacollector = mesa.DataCollector(
            model_reporters={"bottlenecks": "bottleneck_log"},
            agent_reporters={
                "SVI": "svi",
                "status": "status",
                "evacuation_time": "evacuation_time",
                "current_node": "current_pos_node",
            },
        )

        total_time = time.time() - init_start
        print(f"🎉 Model initialization complete in {total_time:.2f}s")
        print(f"📊 Final stats: {len(self.agents):,} agents, {failed_agents} failed")

    def _precompute_shelter_nodes(self, amenities_df: pl.DataFrame) -> Dict[str, set]:
        """OPTIMIZED: Finds nearest graph nodes for all out-of-bounds amenities using OSMnx"""
        shelters = {mode: set() for mode in self.hybrid_managers.keys()}
        if amenities_df.is_empty():
            print("   No amenities data provided")
            return shelters

        print(f"   Processing {len(amenities_df):,} amenities...")

        filter_start = time.time()

        rows_ = []
        for row in amenities_df.iter_rows(named=True):
            if not self.is_pos_in_evacuation_area(
                (row["latitude"], row["longitude"]), False  # TODO:TRY TRUE & FALSE
            ):
                rows_.append(row)

        safe_amenities = pl.DataFrame(rows_)

        del rows_
        gc.collect()
        filter_time = time.time() - filter_start

        print(
            f"   Filtered to {len(safe_amenities):,} safe amenities in {filter_time:.2f}s"
        )

        if len(safe_amenities) == 0:
            return shelters

        # Extract coordinates for batch processing
        amenity_coords = list(
            zip(
                safe_amenities["latitude"].to_list(),
                safe_amenities["longitude"].to_list(),
            )
        )

        # Process each mode in parallel using OSMnx's batch nearest node search
        process_start = time.time()

        for mode, manager in self.hybrid_managers.items():
            if not hasattr(manager, "kdtree") or manager.kdtree is None:
                continue

            # Use OSMnx's efficient nearest node search for all amenities at once
            try:
                # Find nearest nodes for all amenities in one batch
                dists, idxs = manager.kdtree.query(amenity_coords, k=1)
                nearest_nodes = [manager.node_ids[i] for i in idxs]

                # Add to shelters set
                for node in nearest_nodes:
                    if node is not None:
                        shelters[mode].add(node)

                print(f"   Found {len(nearest_nodes):,} shelter nodes for {mode}")
            except Exception as e:
                print(f"   Error processing {mode} shelters: {e}")
                # Fallback to individual processing
                for pos in amenity_coords:
                    node = manager.get_nearest_node(pos)
                    if node is not None:
                        shelters[mode].add(node)

        process_time = time.time() - process_start
        total_shelters = sum(len(s) for s in shelters.values())
        print(f"   Completed processing in {process_time:.2f}s")
        print(f"   Found {total_shelters:,} unique shelter locations across all modes")

        return shelters

    def _precompute_shelter_distance_matrices(self) -> Dict[str, dict]:
        """Precompute distance matrices for all shelter nodes using NetworkX"""
        distance_matrices = {}

        # Calculate total iterations across all modes
        total_iterations = sum(
            len(shelters) for shelters in self.shelter_nodes.values() if shelters
        )

        # Create single progress bar for all iterations
        with tqdm.tqdm(
            total=total_iterations,
            desc="🗺️ Computing Distance Matrices",
            unit="shelter",
            unit_scale=True,
            ncols=120,
            colour="green",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
            ascii=False,
            dynamic_ncols=True,
            smoothing=0.3,
            mininterval=0.1,
            maxinterval=1.0,
            leave=True,
        ) as pbar:

            for mode, shelters in self.shelter_nodes.items():
                if not shelters:
                    continue

                manager = self.hybrid_managers.get(mode)
                if not manager:
                    continue

                # Update postfix to show current mode
                pbar.set_postfix(mode=mode, shelters=f"{len(shelters):,}")
                start_time = time.time()

                try:
                    G = manager.nx_graph
                    distance_matrix = {}
                    shelter_list = list(shelters)

                    # Remove the inner tqdm and just update the main progress bar
                    for i, source in enumerate(shelter_list):
                        if source not in G:
                            pbar.update(1)
                            continue

                        paths = nx.single_source_dijkstra_path_length(
                            G, source, weight="length"
                        )

                        for target in shelter_list:
                            if target in paths:
                                distance_matrix[(source, target)] = paths[target]

                        # Update progress bar for each completed shelter
                        pbar.update(1)

                    distance_matrices[mode] = distance_matrix
                    print(f"     {mode} completed in {time.time() - start_time:.2f}s")

                except Exception as e:
                    print(f"     Error computing distance matrix for {mode}: {e}")
                    distance_matrices[mode] = {}
                    # Update remaining shelters for this mode
                    remaining = (
                        len(shelter_list) - i if "i" in locals() else len(shelter_list)
                    )
                    pbar.update(remaining)

        return distance_matrices

    def get_nearest_node(self, pos: tuple, mode: str) -> Optional[int]:
        """Public interface - uses HybridGraphManager for nearest node lookup."""
        mode = self._normalize_mode(mode)
        manager = self.hybrid_managers.get(mode)
        return manager.get_nearest_node(pos) if manager else None

    def step(self):
        """Advance the model by one time step."""
        step_start = time.time()

        self.edge_load.clear()
        self.edge_agents.clear()
        self.bottleneck_log = []

        self.sim_time += timedelta(seconds=self.step_seconds)

        # Count agent statuses for monitoring
        status_counts = defaultdict(int)
        for agent in self.agents:
            status_counts[agent.status] += 1

        self.agents.shuffle_do("step")
        self._analyze_and_log_bottlenecks()
        self.datacollector.collect(self)

        step_time = time.time() - step_start

        print(
            f"⏰ Step : {step_time:.3f}s | "
            f"Active: {status_counts['EVACUATING']}, "
            f"Planning: {status_counts['PLANNING']}, "
            f"Arrived: {status_counts['ARRIVED']}, "
            f"Failed: {status_counts['FAILED']}, "
            f"Inactive: {status_counts['INACTIVE']}"
        )

    def _analyze_and_log_bottlenecks(self):
        """Iterates through road usage and logs congested edges."""
        for edge, load in self.edge_load.items():
            u, v = edge
            try:
                # Use CAR manager for edge data
                manager = self.hybrid_managers["CAR"]
                edge_data = manager.get_edge_data(u, v)
                capacity = edge_data.get("capacity", 20) if edge_data else 20

                if load > capacity:
                    avg_svi = sum(a.svi for a in self.edge_agents[edge]) / load
                    self.bottleneck_log.append(
                        {
                            "time": self.sim_time,
                            "edge_nodes": (u, v),
                            "load": load,
                            "capacity": capacity,
                            "congestion_index": load / capacity,
                            "avg_svi_stuck": avg_svi,
                        }
                    )
            except (KeyError, TypeError, ZeroDivisionError):
                continue

    # --- HELPER METHODS (API FOR AGENTS) ---

    def is_pos_in_evacuation_area(self, pos: tuple, if_lat_lon: bool = True) -> bool:
        """Checks if a (lat, lon) point is inside the evacuation polygon."""
        if if_lat_lon:
            return self.evac_polygon.contains(
                Point(pos[1], pos[0])
            )  # Convert (lat, lon) to (lon, lat)

        return self.evac_polygon.contains(Point(pos[0], pos[1]))  # Already (lon, lat)

    def get_nearest_shelter_node(self, source_node: int, mode: str) -> Optional[int]:
        """Finds the closest pre-computed shelter to an agent's current node."""
        mode = self._normalize_mode(mode)
        shelters = self.shelter_nodes.get(mode)
        if not shelters or source_node is None:
            return None

        manager = self.hybrid_managers.get(mode)
        if not manager:
            return None

        # Check if source node is already a shelter
        if source_node in shelters:
            return source_node

        # Use precomputed distance matrix if available
        distance_matrix = self.shelter_distance_matrices.get(mode, {})
        if distance_matrix:
            try:
                # Find the closest shelter using precomputed distances
                min_distance = float("inf")
                nearest_shelter = None

                for shelter in shelters:
                    if (source_node, shelter) in distance_matrix:
                        distance = distance_matrix[(source_node, shelter)]
                        if distance < min_distance:
                            min_distance = distance
                            nearest_shelter = shelter

                if nearest_shelter:
                    return nearest_shelter
            except Exception as e:
                print(f"Error using distance matrix for {mode}: {e}")
                # Fall back to direct computation

        # Fallback: compute distance to each shelter
        try:
            G = manager.nx_graph
            distances = {}

            for shelter in shelters:
                try:
                    path_length = nx.shortest_path_length(
                        G, source=source_node, target=shelter, weight="length"
                    )
                    distances[shelter] = path_length
                except nx.NetworkXNoPath:
                    continue

            if not distances:
                return None

            # Return the shelter with minimum distance
            return min(distances.items(), key=lambda x: x[1])[0]

        except Exception as e:
            print(f"Error finding nearest shelter for {mode}: {e}")
            return None

    def plan_route(
        self, agent: EvacuationAgent, source_node: int, target_node: int
    ) -> list:
        """Calculates the shortest path using HybridGraphManager, considering dynamic traffic."""
        mode = self._normalize_mode(agent.main_mode)
        manager = self.hybrid_managers.get(mode)
        if not manager or source_node is None or target_node is None:
            return []

        # Check if source and target are the same
        if source_node == target_node:
            return [source_node]

        agent_speed = max(agent.speed_m_s, 0.1)

        # For car agents, apply congestion penalty
        if mode == "CAR":
            # Create a copy of the graph to modify weights temporarily
            graph = manager.nx_graph.copy()

            # Apply dynamic weights based on congestion
            for u, v, data in graph.edges(data=True):
                try:
                    # Safely convert length to float
                    base_length = float(data.get("length", 1.0))
                except (ValueError, TypeError):
                    base_length = 1.0

                base_travel_time = base_length / agent_speed

                # Get congestion for this edge
                congestion = self.get_edge_congestion((u, v))
                penalty_factor = 2**congestion  # Exponential penalty
                data["dynamic_weight"] = base_travel_time * penalty_factor

            weight = "dynamic_weight"
        else:
            graph = manager.nx_graph
            weight = "length"

        gc.collect()
        try:
            return nx.shortest_path(
                graph, source=source_node, target=target_node, weight=weight
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def report_edge_usage(self, agent: EvacuationAgent, edge: tuple):
        """Called by agents to report which road they are on this step."""
        if agent.main_mode == "CAR":
            self.edge_load[edge] += 1
            self.edge_agents[edge].append(agent)

    def get_edge_congestion(self, edge: tuple) -> float:
        """Returns the congestion index for a given edge from the last step."""
        load = self.edge_load.get(edge, 0)
        if load == 0:
            return 0.0

        try:
            u, v = edge
            manager = self.hybrid_managers["CAR"]
            edge_data = manager.get_edge_data(u, v)
            capacity = edge_data.get("capacity", 20) if edge_data else 20
            return load / capacity
        except (KeyError, TypeError, ZeroDivisionError):
            return 0.0

    @staticmethod
    def _normalize_mode(mode: str) -> str:
        """Normalize transportation mode to standard keys."""
        mode = mode.upper()
        mode_mapping = {
            "CAR": "CAR",
            "DRIVE": "CAR",
            "WALKING": "WALKING",
            "WALK": "WALKING",
            "BIKE": "BIKE",
            "BICYCLE": "BIKE",
            "CYCLING": "BIKE",
        }
        return mode_mapping.get(mode, "WALKING")
