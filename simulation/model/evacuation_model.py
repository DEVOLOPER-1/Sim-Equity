# FILE: evacuation_model.py
# -----------------------------
# This module defines the core agent-based model for the evacuation simulation.
# It has been refactored for performance, clarity, and correctness, using rustworkx
# for graph operations and a robust state machine for agent behavior.

import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import mesa
import polars as pl
import rustworkx as rx
from shapely.geometry import Point, Polygon
from shapely.strtree import STRtree


# --- AGENT DEFINITION ---


class EvacuationAgent(mesa.Agent):
    """
    Represents a person from the NetMob25 dataset with their own unique
    vulnerability, assets, and behavioral parameters.
    """

    def __init__(self, model: mesa.Model, unique_id: str, **kwargs):
        super().__init__(unique_id, model)
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
                self.start_pos, self.main_mode
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
        self.path = self.model.plan_route(self, self.current_pos_node, self.target_node)

        if self.path and len(self.path) >= 2:
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
        graph = self.model.get_graph_for_mode(self.main_mode)
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
        G_drive: rx.PyDiGraph,
        G_walk: rx.PyDiGraph,
        G_cycle: rx.PyDiGraph,
        amenities_df: pl.DataFrame,
        evacuation_area_polygon: Polygon,
        start_datetime: datetime,
        step_seconds: int = 60,
        svi_speed_penalty: float = 0.5,
        max_svi_start_delay_s: int = 1800,
        base_patience_s: int = 300,
    ):
        super().__init__()
        self.start_datetime = start_datetime
        self.sim_time = start_datetime
        self.step_seconds = step_seconds

        # Simulation behavioral parameters
        self.svi_speed_penalty = svi_speed_penalty
        self.max_svi_start_delay_s = max_svi_start_delay_s
        self.base_patience_s = base_patience_s

        # Environment data
        self.evac_polygon = evacuation_area_polygon
        self.graphs = {"DRIVE": G_drive, "WALKING": G_walk, "BIKE": G_cycle}

        # Pre-build spatial indexes for fast nearest-node lookups
        self._build_spatial_indexes()

        # Pre-compute and cache the locations of safe shelters
        self.shelter_nodes = self._precompute_shelter_nodes(amenities_df)

        # Agent scheduling
        for agent_data in agents_df.iter_rows(named=True):
            agent_id = agent_data.get("ID")
            if agent_id:
                EvacuationAgent(model=self, unique_id=str(agent_id), **agent_data)

        # Bottleneck monitoring and data collection
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
        print(f"Model instantiated with {self.agents.__len__()} agents.")

    def _build_spatial_indexes(self):
        """Creates high-performance STRtree spatial indexes for each network graph."""
        self.spatial_indexes = {}
        self.node_id_maps = {}  # Maps STRtree index back to rustworkx node index
        for mode, graph in self.graphs.items():
            points = [Point(data["x"], data["y"]) for _, data in graph.nodes()]
            self.spatial_indexes[mode] = STRtree(points)
            # Create a mapping from the STRtree's sequential index to the actual node ID
            self.node_id_maps[mode] = {
                i: node_id for i, node_id in enumerate(graph.node_indices())
            }
        print("Spatial indexes built successfully.")

    def _precompute_shelter_nodes(self, amenities_df: pl.DataFrame) -> Dict[str, set]:
        """Finds nearest graph nodes for all out-of-bounds amenities in parallel."""
        shelters = {mode: set() for mode in self.graphs.keys()}
        if amenities_df.is_empty():
            return shelters

        # Filter amenities to only those outside the evacuation zone
        safe_amenities = amenities_df.filter(
            ~pl.struct(["latitude", "longitude"]).map_elements(
                lambda pos: self.is_pos_in_evacuation_area(
                    (pos["latitude"], pos["longitude"])
                ),
                return_dtype=pl.Boolean,
            )
        )

        tasks = [
            (row["latitude"], row["longitude"], mode)
            for row in safe_amenities.iter_rows(named=True)
            for mode in shelters.keys()
        ]

        with ThreadPoolExecutor() as executor:
            results = list(
                executor.map(lambda p: self.get_nearest_node(p[:2], p[2]), tasks)
            )

        for i, node in enumerate(results):
            if node is not None:
                _, _, mode = tasks[i]
                shelters[mode].add(node)

        print(
            f"Pre-computed {sum(len(s) for s in shelters.values())} shelter node locations."
        )
        return shelters

    def step(self):
        """Advance the model by one time step."""
        self.edge_load.clear()
        self.edge_agents.clear()
        self.bottleneck_log = []

        self.sim_time += timedelta(seconds=self.step_seconds)
        self.agents.shuffle_do("step")
        self._analyze_and_log_bottlenecks()
        self.datacollector.collect(self)

    def _analyze_and_log_bottlenecks(self):
        """Iterates through road usage and logs congested edges."""
        for edge, load in self.edge_load.items():
            u, v = edge
            try:
                edge_data = self.graphs["DRIVE"].get_edge_data(u, v)
                capacity = edge_data.get("capacity", 20)  # Default capacity

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

    def get_graph_for_mode(self, mode: str) -> rx.PyDiGraph:
        return self.graphs.get(mode.upper(), self.graphs["WALKING"])

    def get_nearest_node(self, pos: tuple, mode: str) -> Optional[int]:
        """Finds the nearest node on the appropriate graph for a given lat/lon."""
        if pos is None:
            return None
        mode = mode.upper()
        point = Point(pos[1], pos[0])  # Shapely uses (lon, lat)

        try:
            tree_index = self.spatial_indexes[mode].nearest(point)
            return self.node_id_maps[mode][tree_index]
        except (KeyError, IndexError):
            return None

    def is_pos_in_evacuation_area(self, pos: tuple) -> bool:
        """Checks if a (lat, lon) point is inside the evacuation polygon."""
        return self.evac_polygon.contains(Point(pos[1], pos[0])) if pos else False

    def get_nearest_shelter_node(self, source_node: int, mode: str) -> Optional[int]:
        """Finds the closest pre-computed shelter to an agent's current node."""
        mode = mode.upper()
        shelters = self.shelter_nodes.get(mode)
        if not shelters or source_node is None:
            return None

        graph = self.get_graph_for_mode(mode)

        # This can still be slow. A better approach for massive simulations
        # would be to pre-calculate paths from all nodes, but this is a robust compromise.
        try:
            paths = rx.dijkstra_shortest_paths(
                graph,
                source=source_node,
                target=list(shelters),
                weight_fn=lambda e: float(e.get("length", 1.0)),
            )
            # Find the target with the shortest path among all reachable shelters
            reachable_shelters = {
                target: path for target, path in paths.items() if path
            }
            if not reachable_shelters:
                return None

            closest_target = min(
                reachable_shelters, key=lambda t: len(reachable_shelters[t])
            )
            return closest_target
        except (rx.NoPathFound, KeyError):
            return None

    def plan_route(
        self, agent: EvacuationAgent, source_node: int, target_node: int
    ) -> list:
        """Calculates the shortest path using Dijkstra, considering dynamic traffic."""
        graph = self.get_graph_for_mode(agent.main_mode)
        agent_speed = max(agent.speed_m_s, 0.1)

        def dynamic_weight_fn(edge_data: dict) -> float:
            """A custom weight function for pathfinding that represents travel time."""
            base_travel_time = edge_data.get("length", 1.0) / agent_speed

            # Only apply congestion penalty for cars
            if agent.main_mode == "CAR":
                u, v = edge_data["source"], edge_data["target"]
                congestion = self.get_edge_congestion((u, v))
                penalty_factor = 2**congestion  # Exponential penalty
                return base_travel_time * penalty_factor

            return base_travel_time

        try:
            path = rx.dijkstra_shortest_paths(
                graph,
                source=source_node,
                target=target_node,
                weight_fn=dynamic_weight_fn,
            )
            return list(path)
        except (rx.NoPathFound, KeyError):
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
            edge_data = self.graphs["DRIVE"].get_edge_data(u, v)
            capacity = edge_data.get("capacity", 20)
            return load / capacity
        except (KeyError, TypeError, ZeroDivisionError):
            return 0.0
