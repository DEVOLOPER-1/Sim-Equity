# FILE: evacuation_model.py
# -----------------------------
# This module defines the core agent-based model for the evacuation simulation,
# using the Mesa framework for agent scheduling and interaction.
import gc
import warnings
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import mesa
import networkx as nx
import osmnx as ox
import polars as pl
import rustworkx as rx
import shapely.geometry
from shapely.geometry import Point


# --- AGENT DEFINITION ---
# (The EvacuationAgent class is unchanged)
class EvacuationAgent(mesa.Agent):
    """
    An individual agent in the evacuation simulation.
    Represents a person with unique vulnerability, assets, and behavioral parameters.
    """

    def __init__(self, model: mesa.Model, original_id: str = None, **kwargs):
        """
        Initialize an agent with properties derived from the initializer CSV.
        Args:
            model (mesa.Model): The parent model instance.
            original_id (str): The original 'ID' of the agent from your data.
            **kwargs: A dictionary of agent properties from the input CSV.
        """
        # MESA 3.0+: No unique_id parameter - automatically assigned
        super().__init__(model)

        # Store your original ID
        self.original_id = original_id
        self.status = "INACTIVE"
        self.svi = float(kwargs.get("SVI_normalized", 0.0))
        self.main_mode = kwargs.get("main_mode", "WALKING")

        # Handle home location coordinates - check for None values
        home_lat = kwargs.get("home_location_lat")
        home_lon = kwargs.get("home_location_lon")
        self.home_location = (
            (float(home_lat), float(home_lon))
            if home_lat is not None and home_lon is not None
            else None
        )

        # --- Position & Routing State ---
        start_lat = kwargs.get("start_lat")
        start_lon = kwargs.get("start_lon")
        if start_lat is not None and start_lon is not None:
            self.start_pos = (float(start_lat), float(start_lon))
        else:
            warnings.warn(f"Agent {original_id}: No valid start position.")
            self.status = "FAILED"
            self.start_pos = None

        self.current_pos_node = None
        if self.status != "FAILED":
            self.current_pos_node = self.model.get_nearest_node(
                self.start_pos, self.main_mode
            )
            if self.current_pos_node is None:
                warnings.warn(
                    f"Agent {original_id}: Could not find a valid starting node for {self.start_pos}."
                )
                self.status = "FAILED"

        self.target_node = None
        self.path: List[Any] = []
        self.current_edge: Optional[tuple] = None
        self.edge_progress_m = 0.0

        # --- Time & Deadline ---
        start_time_str = kwargs.get("start_time")
        try:
            self.initial_activation_time = (
                datetime.fromisoformat(start_time_str)
                if start_time_str
                else model.start_datetime
            )
        except ValueError:
            self.initial_activation_time = model.start_datetime

        self.evacuation_time = 0
        start_delay_s = self.svi * self.model.max_svi_start_delay_s
        self.effective_activation_time = self.initial_activation_time + timedelta(
            seconds=start_delay_s
        )

        base_speed_m_s = self._get_base_speed(kwargs)
        self.speed_m_s = base_speed_m_s * (
            1.0 - self.svi * self.model.svi_speed_penalty
        )
        self.patience_threshold_s = self.model.base_patience_s * (1.0 - self.svi)
        self.time_stuck_s = 0

    def _get_base_speed(self, kwargs: dict) -> float:
        """Helper to get the appropriate speed from the agent's data."""
        if self.main_mode == "WALKING":
            return float(kwargs.get("walking_speed_m_s", 1.4))
        elif self.main_mode == "BIKE":
            return float(kwargs.get("cycling_speed_m_s", 4.5))
        else:
            speed = (
                kwargs.get("median_speed_m_s") or kwargs.get("mean_speed_m_s") or 8.3
            )
            return float(speed)

    def step(self):
        if self.status in ["ARRIVED", "FAILED"]:
            return

        if self.status == "INACTIVE":
            if self.model.sim_time >= self.effective_activation_time:
                self.status = "PLANNING"
            else:
                return

        # Increment evacuation timer only for active agents
        self.evacuation_time += self.model.step_seconds

        # --- State: PLANNING ---
        # The agent decides where to go and calculates its initial route.
        # --- State: PLANNING ---
        # The agent decides where to go and calculates its initial route.
        if self.status == "PLANNING":
            self.plan_evacuation_route()

        # --- State: EVACUATING ---
        if self.status == "EVACUATING":
            # Check if stuck in traffic and if patience has run out.
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
        """Determines agent's destination and calculates the path."""
        # Skip planning if agent is already failed
        if self.status == "FAILED":
            return

        # Destination Logic: Is home a safe option?
        target_node = None
        if self.home_location and not self.model.is_pos_in_evacuation_area(
            self.home_location
        ):
            target_node = self.model.get_nearest_node(
                self.home_location, self.main_mode
            )

        if target_node is None:
            target_node = self.model.get_nearest_shelter_node(
                self.current_pos_node, self.main_mode
            )

        self.target_node = target_node
        if self.target_node is None:
            warnings.warn(
                f"Agent {self.original_id}: Could not find any target destination."
            )
            self.status = "FAILED"
            return

        self.path = self.model.plan_route(self, self.current_pos_node, self.target_node)

        if self.path and len(self.path) >= 2:
            self.status = "EVACUATING"
            self.current_edge = None
        else:
            warnings.warn(
                f"Agent {self.original_id}: Could not find path from {self.current_pos_node} to {self.target_node}."
            )
            self.status = "FAILED"

    def move(self):
        if not self.path or len(self.path) < 2:
            self.status = (
                "ARRIVED" if self.current_pos_node == self.target_node else "FAILED"
            )
            return

        # Set current edge if we are starting a new one
        if self.current_edge is None:
            self.current_edge = (self.path[0], self.path[1])
            self.edge_progress_m = 0.0

        # Calculate distance to travel in this step
        distance_to_travel = self.speed_m_s * self.model.step_seconds

        # Traverse the path graph
        while distance_to_travel > 0:
            if not self.current_edge:
                break
            u, v = self.current_edge
            try:
                # Assuming key=0 for simplicity in multigraphs
                edge_data = self.model.get_graph_for_mode(self.main_mode).get_edge_data(
                    u, v, 0
                )
                edge_length = edge_data["length"]
            except (KeyError, TypeError):
                warnings.warn(
                    f"Agent {self.original_id}: Edge ({u}, {v}) not found in graph."
                )
                self.status = "FAILED"
                return

            # OSMnx graphs can have parallel edges, so we get the first one (key=0)
            edge_length = edge_data[0]["length"]

            remaining_on_edge = edge_length - self.edge_progress_m

            if distance_to_travel >= remaining_on_edge:
                # We will complete this edge and maybe move to the next
                self.current_pos_node = v
                self.path.pop(0)
                distance_to_travel -= remaining_on_edge

                # Report usage of the *completed* edge
                self.model.report_edge_usage(self, (u, v, 0))

                if len(self.path) < 2:
                    self.status = "ARRIVED"
                    self.current_edge = None
                    break
                else:
                    self.current_edge = (self.path[0], self.path[1])
                    self.edge_progress_m = 0.0
            else:
                # Move partially along the current edge
                self.edge_progress_m += distance_to_travel
                distance_to_travel = 0

                # Report usage of the *current* edge
                self.model.report_edge_usage(self, (u, v, 0))

    def is_stuck_in_traffic(self) -> bool:
        """Checks if the agent's current path is congested."""
        if not self.current_edge or self.main_mode != "CAR":
            return False
        return self.model.get_edge_congestion(self.current_edge) > 1.0


class EvacuationModel(mesa.Model):
    """
    The main model class for the evacuation simulation.
    It holds the environment, manages agents, and collects data.
    """

    def __init__(
        self,
        agents_df: pl.DataFrame,
        G_drive: nx.MultiDiGraph,
        G_walk: nx.MultiDiGraph,
        G_cycle: nx.MultiDiGraph,
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

        # --- Simulation Parameters ---
        # These parameters can be adjusted to run different experimental scenarios.
        self.svi_speed_penalty = svi_speed_penalty
        self.max_svi_start_delay_s = max_svi_start_delay_s
        self.base_patience_s = base_patience_s

        # Keep NetworkX graphs as the source of truth for compatibility with OSMnx
        self.G_drive = G_drive
        self.G_walk = G_walk
        self.G_cycle = G_cycle
        self.evac_polygon = evacuation_area_polygon

        if not amenities_df.is_empty():
            # Use the NetworkX graph for this OSMnx function
            self.amenity_nodes = ox.nearest_nodes(
                self.G_walk, X=amenities_df["longitude"], Y=amenities_df["latitude"]
            )
            if not isinstance(self.amenity_nodes, list):
                self.amenity_nodes = [self.amenity_nodes]
        else:
            print("No amenities provided!")
            self.amenity_nodes = []

        # --- OPTIMIZATION: Create persistent rustworkx graphs for routing ---
        print("Converting NetworkX graphs to rustworkx for performance...")
        self.rx_drive, self.drive_nx_to_rx, self.drive_rx_to_nx = (
            self._create_persistent_rx_graph(self.G_drive)
        )
        self.rx_walk, self.walk_nx_to_rx, self.walk_rx_to_nx = (
            self._create_persistent_rx_graph(self.G_walk)
        )
        self.rx_cycle, self.cycle_nx_to_rx, self.cycle_rx_to_nx = (
            self._create_persistent_rx_graph(self.G_cycle)
        )

        # Store edge weight mappings for congestion-aware routing
        self.drive_edge_weights = {}  # Maps (u, v, key) to current weight
        self._initialize_edge_weights()
        print("Graph conversion complete.")

        for agent_data in agents_df.iter_rows(named=True):
            EvacuationAgent(model=self, original_id=agent_data.get("ID"), **agent_data)

        print(f"Created {len(self.agents)} agents successfully.")

        # --- Data Collection & Bottleneck Monitoring ---
        self.edge_load = defaultdict(int)
        self.edge_agents = defaultdict(list)
        self.bottleneck_log: List[Dict[str, Any]] = []

        # MESA 3.0+: DataCollector is now in mesa.datacollection
        self.datacollector = mesa.datacollection.DataCollector(
            model_reporters={"bottlenecks": "bottleneck_log"},
            agent_reporters={
                "SVI": "svi",
                "status": "status",
                "evacuation_time": "evacuation_time",
                "current_node": "current_pos_node",
            },
        )

    @staticmethod
    def _create_persistent_rx_graph(nx_graph: nx.MultiDiGraph):
        """Converts a NetworkX graph to a rustworkx graph and creates node mappings."""
        rx_graph = rx.PyDiGraph()
        nx_to_rx_map = {}
        rx_to_nx_map = []

        for node in nx_graph.nodes():
            new_index = rx_graph.add_node(None)
            nx_to_rx_map[node] = new_index
            rx_to_nx_map.append(node)

        for u, v, key, data in nx_graph.edges(data=True, keys=True):
            source_idx = nx_to_rx_map[u]
            target_idx = nx_to_rx_map[v]
            # Store original length and other data in the edge payload
            rx_graph.add_edge(source_idx, target_idx, data.copy())

        return rx_graph, nx_to_rx_map, rx_to_nx_map

    def _initialize_edge_weights(self):
        """Initialize edge weights for congestion calculations."""
        for u, v, key, data in self.G_drive.edges(data=True, keys=True):
            base_length = data.get("length", 1.0)
            self.drive_edge_weights[(u, v, key)] = base_length

    @staticmethod
    def _get_drive_weight_function():
        """Returns a weight function for driving that accounts for congestion."""

        def weight_fn(edge_data):
            # edge_data is the data dictionary stored on the rustworkx edge
            base_length = edge_data.get("length", 1.0)

            # For simplicity, we'll use base length here
            # In practice, you'd want to look up current congestion
            return base_length

        return weight_fn

    def step(self):
        self._update_congestion_weights()
        self.edge_load.clear()
        self.edge_agents.clear()
        self.bottleneck_log = []

        # 2. Advance the simulation clock
        self.sim_time += timedelta(seconds=self.step_seconds)

        # 3. MESA 3.0+: Use AgentSet methods to activate agents
        # RandomActivation equivalent: shuffle agents and call step() on each
        self.agents.shuffle_do("step")

        # 4. Analyze traffic and log bottlenecks after all agents have moved
        self._analyze_and_log_bottlenecks()

        # 5. Collect data for this step
        self.datacollector.collect(self)

    def _update_congestion_weights(self):
        """Updates edge weights based on current congestion for the driving graph."""
        for (u, v, key), load in self.edge_load.items():
            if (u, v, key) in self.drive_edge_weights:
                # Get capacity from the original NetworkX graph
                capacity = self.G_drive.edges[u, v, key].get("capacity", 20)
                congestion = load / capacity if capacity > 0 else float("inf")

                base_length = self.G_drive.edges[u, v, key].get("length", 1.0)
                weight = max(0.001, base_length * (1.0 + congestion))
                self.drive_edge_weights[(u, v, key)] = weight

    def _analyze_and_log_bottlenecks(self):
        """Iterates through road usage and logs congested edges."""
        for edge, load in self.edge_load.items():
            try:
                # Capacity must be pre-calculated and added to your graph edges
                # A simple heuristic: capacity = number_of_lanes * vehicles_per_minute
                u, v, key = edge
                capacity = self.G_drive.edges[u, v, key].get("capacity", 20)
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

    def get_graph_for_mode(self, main_mode: str) -> nx.MultiDiGraph:
        """Returns the NetworkX graph for compatibility with OSMnx and attribute lookups."""
        if main_mode == "WALKING":
            return self.G_walk
        if main_mode == "BIKE":
            return self.G_cycle
        else:  # CAR, PT, etc. all use the drive network
            return self.G_drive

    def get_nearest_node(self, pos: tuple, mode: str) -> Optional[int]:
        """Finds the nearest node using OSMnx, which requires a NetworkX graph."""
        if pos is None or len(pos) != 2:
            return None
        lat, lon = pos
        try:
            # This function requires a NetworkX graph
            return ox.distance.nearest_nodes(
                self.get_graph_for_mode(mode), X=lon, Y=lat
            )
        except Exception as e:
            warnings.warn(f"Error finding nearest node for pos {pos}: {e}")
            return None

    def is_pos_in_evacuation_area(self, pos: tuple) -> bool:
        if pos is None or len(pos) != 2:
            return False
        return self.evac_polygon.contains(Point(pos[1], pos[0]))

    def get_nearest_shelter_node(self, source_node: int, mode: str) -> Optional[int]:
        """Finds the nearest shelter using the high-performance rustworkx graph."""
        if not source_node or not self.amenity_nodes:
            return None

        # Select the correct pre-built rustworkx graph and mappings
        if mode == "WALKING":
            rx_graph, nx_to_rx, rx_to_nx = (
                self.rx_walk,
                self.walk_nx_to_rx,
                self.walk_rx_to_nx,
            )
            weight_fn = lambda edge_data: edge_data.get("length", 1.0)
        elif mode == "BIKE":
            rx_graph, nx_to_rx, rx_to_nx = (
                self.rx_cycle,
                self.cycle_nx_to_rx,
                self.cycle_rx_to_nx,
            )
            weight_fn = lambda edge_data: edge_data.get("length", 1.0)
        else:  # CAR
            rx_graph, nx_to_rx, rx_to_nx = (
                self.rx_drive,
                self.drive_nx_to_rx,
                self.drive_rx_to_nx,
            )
            weight_fn = self._get_drive_weight_function()

        if source_node not in nx_to_rx:
            return None
        source_index = nx_to_rx[source_node]

        # Calculate distances to all nodes from the source at once
        distances = rx.dijkstra_shortest_path_lengths(
            rx_graph, node=source_index, edge_cost_fn=weight_fn
        )

        min_dist = float("inf")
        closest_shelter = None
        for shelter_node in self.amenity_nodes:
            if shelter_node in nx_to_rx:
                shelter_index = nx_to_rx[shelter_node]
                dist = distances.get(shelter_index)
                if dist is not None and dist < min_dist:
                    min_dist = dist
                    closest_shelter = shelter_node
        gc.collect()
        return closest_shelter

    def plan_route(
        self, agent: EvacuationAgent, source_node: Any, target_node: Any
    ) -> List[Any]:
        """Plans a route using the pre-built, congestion-aware rustworkx graph."""
        if agent.main_mode == "WALKING":
            rx_graph, nx_to_rx, rx_to_nx = (
                self.rx_walk,
                self.walk_nx_to_rx,
                self.walk_rx_to_nx,
            )
            weight_fn = lambda edge_data: edge_data.get("length", 1.0)
        elif agent.main_mode == "BIKE":
            rx_graph, nx_to_rx, rx_to_nx = (
                self.rx_cycle,
                self.cycle_nx_to_rx,
                self.cycle_rx_to_nx,
            )
            weight_fn = lambda edge_data: edge_data.get("length", 1.0)
        else:  # CAR
            rx_graph, nx_to_rx, rx_to_nx = (
                self.rx_drive,
                self.drive_nx_to_rx,
                self.drive_rx_to_nx,
            )
            weight_fn = self._get_drive_weight_function()

        if source_node not in nx_to_rx or target_node not in nx_to_rx:
            warnings.warn(
                f"Source {source_node} or Target {target_node} not in graph map."
            )
            return []

        source_index, target_index = nx_to_rx[source_node], nx_to_rx[target_node]

        # Use dijkstra_shortest_paths with proper parameters
        try:
            paths = rx.dijkstra_shortest_paths(
                rx_graph, source=source_index, target=target_index, weight_fn=weight_fn
            )

            # paths is a dictionary with target_index as key
            path_indices = paths.get(target_index)
            if path_indices:
                return [rx_to_nx[i] for i in path_indices]
            else:
                warnings.warn(
                    f"No path found from {source_node} to {target_node} for agent {agent.original_id}."
                )
                return []
        except Exception as e:
            warnings.warn(f"Error in pathfinding for agent {agent.original_id}: {e}")
            return []

    def report_edge_usage(self, agent: EvacuationAgent, edge: tuple):
        self.edge_load[edge] += 1
        self.edge_agents[edge].append(agent)

    def get_edge_congestion(self, edge: tuple) -> float:
        """Returns the congestion index for a given edge from the last step."""
        try:
            load = self.edge_load.get(edge, 0)
            u, v, key = edge
            capacity = self.G_drive.edges[u, v, key].get("capacity", 20)
            return load / capacity if capacity > 0 else float("inf")
        except KeyError:
            return 0.0
