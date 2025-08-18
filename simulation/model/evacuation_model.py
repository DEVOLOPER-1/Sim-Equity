# FILE: evacuation_model.py
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

import mesa
import polars as pl
import rustworkx as rx
import shapely.geometry
from shapely.geometry import Point
from shapely.strtree import STRtree


# --- AGENT DEFINITION ---
class EvacuationAgent(mesa.Agent):
    """Represents a person with unique vulnerability and behavioral parameters."""

    def __init__(self, model: mesa.Model, original_id: str = None, **kwargs):
        super().__init__(model)
        self.time_stuck_s = None
        self.original_id = original_id
        self.status = "INACTIVE"
        self.svi = float(kwargs.get("SVI_normalized", 0.0))
        self.main_mode = kwargs.get("main_mode", "WALKING")

        # Location handling
        self.home_location = self._get_location(
            kwargs, "home_location_lat", "home_location_lon"
        )
        self.start_pos = self._get_location(kwargs, "start_lat", "start_lon")

        self.current_pos_node = None
        if self.start_pos:
            self.current_pos_node = self.model.get_nearest_node(
                self.start_pos, self.main_mode
            )
            if self.current_pos_node is None:
                warnings.warn(f"Agent {original_id}: No valid starting node")
                self.status = "FAILED"

        # Routing state
        self.target_node = None
        self.path: List[Any] = []
        self.current_edge: Optional[tuple] = None
        self.edge_progress_m = 0.0

        # Time management
        self._init_timing(kwargs)

    @staticmethod
    def _get_location(
        data: dict, lat_key: str, lon_key: str
    ) -> Optional[Tuple[float, float]]:
        """Extract and validate location coordinates."""
        lat = data.get(lat_key)
        lon = data.get(lon_key)
        if lat is not None and lon is not None:
            try:
                return (float(lat), float(lon))
            except ValueError:
                pass
        return None

    def _init_timing(self, kwargs: dict):
        """Initialize agent timing parameters."""
        start_time_str = kwargs.get("start_time")
        try:
            self.initial_activation_time = (
                datetime.fromisoformat(start_time_str)
                if start_time_str
                else self.model.start_datetime
            )
        except ValueError:
            self.initial_activation_time = self.model.start_datetime

        start_delay_s = self.svi * self.model.max_svi_start_delay_s
        self.effective_activation_time = self.initial_activation_time + timedelta(
            seconds=start_delay_s
        )
        self.evacuation_time = 0

        # Movement parameters
        base_speed_m_s = self._get_base_speed(kwargs)
        self.speed_m_s = base_speed_m_s * (
            1.0 - self.svi * self.model.svi_speed_penalty
        )
        self.patience_threshold_s = self.model.base_patience_s * (1.0 - self.svi)
        self.time_stuck_s = 0

    def _get_base_speed(self, kwargs: dict) -> float:
        """Get mode-specific base speed."""
        mode_speeds = {
            "WALKING": kwargs.get("walking_speed_m_s", 1.4),
            "BIKE": kwargs.get("cycling_speed_m_s", 4.5),
        }
        return float(
            mode_speeds.get(self.main_mode, kwargs.get("median_speed_m_s", 8.3))
        )

    def step(self):
        """Agent behavior per simulation step."""
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
                    self.status = "PLANNING"
                    self.time_stuck_s = 0
                    return
            else:
                self.time_stuck_s = 0
            self.move()

    def plan_evacuation_route(self):
        """Determine destination and calculate route."""
        if self.status == "FAILED":
            return

        # Try home location first if safe
        if self.home_location and not self.model.is_pos_in_evacuation_area(
            self.home_location
        ):
            self.target_node = self.model.get_nearest_node(
                self.home_location, self.main_mode
            )

        # Fallback to shelter
        if self.target_node is None:
            self.target_node = self.model.get_nearest_shelter_node(
                self.current_pos_node, self.main_mode
            )

        if self.target_node is None:
            warnings.warn(f"Agent {self.original_id}: No valid destination")
            self.status = "FAILED"
            return

        # Plan route
        self.path = self.model.plan_route(
            self.main_mode, self.current_pos_node, self.target_node
        )

        if self.path and len(self.path) >= 2:
            self.status = "EVACUATING"
            self.current_edge = None
        else:
            warnings.warn(f"Agent {self.original_id}: Path not found")
            self.status = "FAILED"

    def move(self):
        """Move agent along the current path."""
        if not self.path or len(self.path) < 2:
            self.status = (
                "ARRIVED" if self.current_pos_node == self.target_node else "FAILED"
            )
            return

        # Initialize new edge
        if self.current_edge is None:
            self.current_edge = (self.path[0], self.path[1])
            self.edge_progress_m = 0.0

        # Get edge data
        try:
            graph = self.model.get_graph_for_mode(self.main_mode)
            u, v = self.current_edge
            edge_data = graph.get_edge_data(u, v)
            edge_length = edge_data.get("length", 1.0)
        except (KeyError, TypeError):
            warnings.warn(f"Agent {self.original_id}: Edge data missing")
            self.status = "FAILED"
            return

        distance_to_travel = self.speed_m_s * self.model.step_seconds

        while distance_to_travel > 0 and self.current_edge:
            remaining_on_edge = edge_length - self.edge_progress_m

            if distance_to_travel >= remaining_on_edge:
                # Complete current edge
                self.current_pos_node = v
                self.path.pop(0)
                distance_to_travel -= remaining_on_edge
                self.model.report_edge_usage(self, (u, v))

                if len(self.path) < 2:
                    self.status = "ARRIVED"
                    self.current_edge = None
                    break
                else:
                    # Start next edge
                    self.current_edge = (self.path[0], self.path[1])
                    self.edge_progress_m = 0.0
                    try:
                        edge_data = graph.get_edge_data(*self.current_edge)
                        edge_length = edge_data.get("length", 1.0)
                    except (KeyError, TypeError):
                        warnings.warn(f"Agent {self.original_id}: Next edge missing")
                        self.status = "FAILED"
                        break
            else:
                # Partial progress on current edge
                self.edge_progress_m += distance_to_travel
                distance_to_travel = 0
                self.model.report_edge_usage(self, (u, v))

    def is_stuck_in_traffic(self) -> bool:
        """Check if agent is in congested traffic."""
        return (
            self.main_mode == "CAR"
            and self.current_edge
            and self.model.get_edge_congestion(self.current_edge) > 1.0
        )


class EvacuationModel(mesa.Model):
    """Core model for evacuation simulation using rustworkx with optimized spatial indexing."""

    def __init__(
        self,
        agents_df: pl.DataFrame,
        G_drive: rx.PyDiGraph,
        G_walk: rx.PyDiGraph,
        G_cycle: rx.PyDiGraph,
        amenities_df: pl.DataFrame,
        evacuation_area_polygon: shapely.geometry.Polygon,
        start_datetime: datetime,
        step_seconds: int = 60,
        svi_speed_penalty: float = 0.5,
        max_svi_start_delay_s: int = 1200,
        base_patience_s: int = 300,
    ):
        super().__init__()
        self.start_datetime = start_datetime
        self.sim_time = start_datetime
        self.step_seconds = step_seconds

        # Simulation parameters
        self.svi_speed_penalty = svi_speed_penalty
        self.max_svi_start_delay_s = max_svi_start_delay_s
        self.base_patience_s = base_patience_s

        # Spatial data
        self.evac_polygon = evacuation_area_polygon

        # Graph networks
        self.graphs = {"DRIVE": G_drive, "WALKING": G_walk, "BIKE": G_cycle}

        # Create spatial indexes for each graph
        self.spatial_indexes = {}
        self.node_points = {}  # Store points for each mode
        self.node_indices = {}  # Store node indices for each mode

        for mode, graph in self.graphs.items():
            points = []
            node_ids = []

            for node_idx in graph.node_indices():
                node_data = graph[node_idx]
                point = None
                if "x" in node_data and "y" in node_data:
                    point = Point(node_data["x"], node_data["y"])
                elif "coords" in node_data:
                    point = Point(node_data["coords"])

                if point:
                    points.append(point)
                    node_ids.append(node_idx)

            self.spatial_indexes[mode] = STRtree(points)
            self.node_points[mode] = points
            self.node_indices[mode] = node_ids

        # Precompute shelter nodes in parallel
        self.shelter_nodes = self._precompute_shelter_nodes(amenities_df)

        # Initialize drive edge weights in the graph itself
        self._initialize_drive_edge_weights()

        # Create agents
        agent_data_list = list(agents_df.iter_rows(named=True))
        for agent_data in agent_data_list:
            EvacuationAgent(model=self, original_id=agent_data.get("ID"), **agent_data)

        print(f"Created {len(self.agents)} agents successfully. {datetime.now()}")

        # Traffic monitoring (only for drive mode)
        self.edge_load = defaultdict(int)
        self.edge_agents = defaultdict(list)
        self.bottleneck_log: List[Dict[str, Any]] = []

        # Data collection
        self.datacollector = mesa.datacollection.DataCollector(
            model_reporters={"bottlenecks": "bottleneck_log"},
            agent_reporters={
                "SVI": "svi",
                "status": "status",
                "evacuation_time": "evacuation_time",
                "current_node": "current_pos_node",
            },
        )

    def _initialize_drive_edge_weights(self):
        """Initialize edge weights directly in the drive graph."""
        drive_graph = self.graphs["DRIVE"]
        for edge_idx in drive_graph.edge_indices():
            edge_data = drive_graph.get_edge_data_by_index(edge_idx)
            if edge_data is None:
                continue
            base_length = edge_data.get("length", 1.0)
            # Set initial weight to base length
            edge_data["weight"] = base_length

    def _precompute_shelter_nodes(
        self, amenities_df: pl.DataFrame
    ) -> Dict[str, Set[int]]:
        """Find nearest shelter nodes for each mode in parallel."""
        shelters = {mode: set() for mode in ["DRIVE", "WALKING", "BIKE"]}

        if amenities_df.is_empty():
            return shelters

        # Prepare tasks for parallel processing
        tasks = []
        for row in amenities_df.iter_rows(named=True):
            pos = (row["latitude"], row["longitude"])
            if not self.is_pos_in_evacuation_area(pos):
                for mode in shelters.keys():
                    tasks.append((pos, mode))

        # Find nearest nodes in parallel
        with ThreadPoolExecutor(max_workers=15) as executor:
            results = list(
                executor.map(
                    lambda args: self.get_nearest_node(args[0], args[1]), tasks
                )
            )

        # Aggregate results
        for (pos, mode), node in zip(tasks, results):
            if node is not None:
                shelters[mode].add(node)

        return shelters

    """
    def _initialize_edge_weights(self):
        '''Initialize edge weights for congestion calculations.'''
        drive_graph = self.graphs["DRIVE"]
        for u, v in drive_graph.edge_list():
            edge_data = drive_graph.get_edge_data(u, v)
            if edge_data is None:
                continue
            base_length = edge_data.get("length", 1.0)
            self.drive_edge_weights[(u, v)] = base_length
        """

    def step(self):
        """Advance model by one step using sequential agent processing."""
        self._update_congestion_weights()
        self.edge_load.clear()
        self.edge_agents.clear()
        self.bottleneck_log = []
        self.sim_time += timedelta(seconds=self.step_seconds)

        # Collect data before stepping agents
        self.datacollector.collect(self)

        # Process all agents
        self.agents.shuffle_do("step")

        # Analyze bottlenecks
        self._analyze_and_log_bottlenecks()

        # Collect data after stepping agents
        self.datacollector.collect(self)

    def _update_congestion_weights(self):
        """Update edge weights in the graph based on current congestion."""
        drive_graph = self.graphs["DRIVE"]
        for edge, load in self.edge_load.items():
            u, v = edge
            edge_indices = drive_graph.edge_indices_from_endpoints(u, v)

            # Handle the case where multiple edges exist between the same nodes
            if not edge_indices:
                continue

            # Option 1: Update all edges between these nodes (recommended)
            for edge_idx in edge_indices:
                edge_data = drive_graph.get_edge_data_by_index(edge_idx)
                if edge_data is None:
                    continue

                capacity = edge_data.get("capacity", 20)
                congestion = load / capacity if capacity > 0 else float("inf")
                base_length = edge_data.get("length", 1.0)
                weight = max(0.001, base_length * (1.0 + congestion))
                # Update weight directly in edge data
                edge_data["weight"] = weight

    def _analyze_and_log_bottlenecks(self):
        """Log congested edges."""
        if not self.edge_load:
            return

        for edge, load in self.edge_load.items():
            try:
                u, v = edge
                graph = self.graphs["DRIVE"]
                edge_data = graph.get_edge_data(u, v)
                if edge_data is None:
                    continue
                capacity = edge_data.get("capacity", 20)

                if 0 < capacity < load:
                    agents_on_edge = self.edge_agents[edge]
                    avg_svi = (
                        sum(a.svi for a in agents_on_edge) / len(agents_on_edge)
                        if agents_on_edge
                        else 0
                    )
                    self.bottleneck_log.append(
                        {
                            "time": self.sim_time,
                            "edge_nodes": f"({u}, {v})",
                            "load": load,
                            "capacity": capacity,
                            "congestion_index": load / capacity,
                            "avg_svi_stuck": avg_svi,
                        }
                    )
            except (KeyError, ZeroDivisionError):
                continue

    def get_graph_for_mode(self, mode: str) -> rx.PyDiGraph:
        """Get graph for transportation mode."""
        mode_key = "DRIVE" if mode not in ["WALKING", "BIKE"] else mode.upper()
        return self.graphs.get(mode_key, self.graphs["DRIVE"])

    def get_nearest_node(self, pos: tuple, mode: str) -> Optional[int]:
        """Find nearest node using spatial index with O(1) lookup."""
        if pos is None or len(pos) != 2:
            return None

        target_point = Point(pos[1], pos[0])  # (lon, lat) format
        spatial_index = self.spatial_indexes.get(mode.upper())
        points = self.node_points.get(mode.upper())
        node_indices = self.node_indices.get(mode.upper())

        if spatial_index is None or not points or not node_indices:
            return None

        # Find nearest point index from spatial index
        nearest_idx = spatial_index.nearest(target_point)
        if nearest_idx is None or nearest_idx >= len(node_indices):
            return None

        # Return the corresponding node index
        return node_indices[nearest_idx]

    def is_pos_in_evacuation_area(self, pos: tuple) -> bool:
        """Check if position is within evacuation zone."""
        if pos is None or len(pos) != 2:
            return False
        return self.evac_polygon.contains(Point(pos[1], pos[0]))

    def get_nearest_shelter_node(self, source_node: int, mode: str) -> Optional[int]:
        """Find nearest shelter node from current position."""
        if source_node is None:
            return None

        mode_key = "DRIVE" if mode not in ["WALKING", "BIKE"] else mode.upper()
        shelters = self.shelter_nodes.get(mode_key, set())
        if not shelters:
            return None

        graph = self.get_graph_for_mode(mode)

        # Compute all distances from source_node
        try:
            all_distances = rx.dijkstra_shortest_path_lengths(
                graph, node=source_node, edge_cost_fn=lambda e: e.get("length", 1.0)
            )
        except (rx.NullGraph, ValueError):
            return None

        # Find closest shelter from the computed distances
        min_distance = float("inf")
        nearest_shelter = None
        for shelter in shelters:
            distance = all_distances.get(shelter, float("inf"))
            if distance < min_distance:
                min_distance = distance
                nearest_shelter = shelter

        return nearest_shelter

    def plan_route(self, mode: str, source_node: int, target_node: int) -> List[int]:
        """Plan route between nodes using rustworkx."""
        if source_node is None or target_node is None or source_node == target_node:
            return []

        graph = self.get_graph_for_mode(mode)

        # Use weight attribute for drive mode, length for others
        if mode != "WALKING" and mode != "BIKE":
            weight_fn = lambda e: float(e.get("weight", 1.0))
        else:
            weight_fn = lambda e: float(e.get("length", 1.0))

        try:
            path_dict = rx.dijkstra_shortest_paths(
                graph, source=source_node, target=target_node, weight_fn=weight_fn
            )
            return path_dict.get(target_node, [])
        except (rx.NullGraph, ValueError):
            return []

    def report_edge_usage(self, agent: EvacuationAgent, edge: tuple):
        """Record edge usage only for drive mode agents."""
        # Only track drive agents (others don't cause congestion)
        if agent.main_mode not in ["WALKING", "BIKE"]:
            self.edge_load[edge] += 1
            self.edge_agents[edge].append(agent)

    def get_edge_congestion(self, edge: tuple) -> float:
        """Calculate current congestion level for an edge."""
        try:
            load = self.edge_load.get(edge, 0)
            u, v = edge
            graph = self.graphs["DRIVE"]
            edge_data = graph.get_edge_data(u, v)
            if edge_data is None:
                return 0.0
            capacity = edge_data.get("capacity", 20)
            return load / capacity if capacity > 0 else float("inf")
        except KeyError:
            return 0.0
