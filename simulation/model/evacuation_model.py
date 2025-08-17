# FILE: evacuation_model.py
# -----------------------------
# This module defines the core agent-based model for the evacuation simulation,
# using the Mesa framework for agent scheduling and interaction.
from collections import defaultdict
from datetime import datetime, timedelta

import mesa
import networkx as nx
import osmnx as ox
import polars as pl
import shapely.geometry
from shapely.geometry import Point


# --- AGENT DEFINITION ---


class EvacuationAgent(mesa.Agent):
    """
    An individual agent in the evacuation simulation.
    Represents a person from the NetMob25 dataset with their own unique
    vulnerability, assets, and behavioral parameters.
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

        # --- Core Properties & State Machine ---
        self.status = "INACTIVE"  # State machine: INACTIVE -> PLANNING -> EVACUATING -> (ARRIVED | FAILED)
        self.svi = float(kwargs.get("SVI_normalized", 0.0))
        self.main_mode = kwargs.get("main_mode", "WALKING")  # Default to walking

        # Handle home location coordinates - check for None values
        home_lat = kwargs.get("home_location_lat")
        home_lon = kwargs.get("home_location_lon")
        if home_lat is not None and home_lon is not None:
            self.home_location = (float(home_lat), float(home_lon))
        else:
            self.home_location = None
            print(f"Agent {original_id}: No valid home location data")

        # --- Position & Routing State ---
        start_lat = kwargs.get("start_lat")
        start_lon = kwargs.get("start_lon")
        if start_lat is not None and start_lon is not None:
            self.start_pos = (float(start_lat), float(start_lon))
        else:
            print(f"Agent {original_id}: No valid start position")
            self.status = "FAILED"
            self.start_pos = None

        if self.status != "FAILED":
            self.current_pos_node = self.model.get_nearest_node(
                self.start_pos, self.main_mode
            )

            # If we can't find a valid starting position, mark agent as failed
            if self.current_pos_node is None:
                print(
                    f"Agent {original_id}: Could not find valid starting node for {self.start_pos}"
                )
                self.status = "FAILED"
                self.current_pos_node = (
                    0  # Set to some default to prevent further errors
                )
        else:
            self.current_pos_node = 0

        self.target_node = None
        self.path = []  # List of network nodes to traverse
        self.current_edge = None
        self.edge_progress_m = 0.0

        # --- Time & Deadline ---
        start_time_str = kwargs.get("start_time")
        if start_time_str:
            try:
                self.initial_activation_time = datetime.fromisoformat(start_time_str)
            except ValueError:
                # Fallback to model start time if parsing fails
                self.initial_activation_time = model.start_datetime
        else:
            self.initial_activation_time = model.start_datetime

        self.evacuation_time = 0  # Seconds since activation

        # --- SVI-driven Behavioral Parameters ---

        # 1. Reaction Delay: Higher SVI means a slower reaction time.
        start_delay_s = self.svi * self.model.max_svi_start_delay_s
        self.effective_activation_time = self.initial_activation_time + timedelta(
            seconds=start_delay_s
        )

        # 2. Speed Penalty: Higher SVI makes the agent move slower.
        # We fetch the base speed appropriate for the agent's main mode.
        base_speed_m_s = self._get_base_speed(kwargs)
        self.speed_m_s = base_speed_m_s * (
            1.0 - self.svi * self.model.svi_speed_penalty
        )

        # 3. Patience/Rerouting: Higher SVI means less patience when stuck.
        self.patience_threshold_s = self.model.base_patience_s * (1.0 - self.svi)
        self.time_stuck_s = 0

    def _get_base_speed(self, kwargs: dict) -> float:
        """Helper to get the appropriate speed from the agent's data."""
        if self.main_mode == "WALKING":
            return kwargs.get("walking_speed_m_s", 1.4)  # Default walking speed 1.4 m/s
        elif self.main_mode == "BIKE":
            return kwargs.get("cycling_speed_m_s", 4.5)  # Default cycling speed 4.5 m/s
        else:  # CAR, PT, etc.
            # Try different speed columns in order of preference
            speed = (
                kwargs.get("median_speed_m_s") or kwargs.get("mean_speed_m_s") or 8.3
            )  # Default car speed 30 km/h
            return float(speed)

    def step(self):
        """The main logic loop for the agent, executed at each model step."""
        if self.status in ["ARRIVED", "FAILED"]:
            return  # Agent's simulation is complete.

        # --- State: INACTIVE ---
        # An agent does nothing until the simulation time passes their activation time.
        if self.status == "INACTIVE":
            if self.model.sim_time >= self.effective_activation_time:
                self.status = "PLANNING"
            else:
                return

        # Increment evacuation timer only for active agents
        self.evacuation_time += self.model.step_seconds

        # --- State: PLANNING ---
        # The agent decides where to go and calculates its initial route.
        if self.status == "PLANNING":
            self.plan_evacuation_route()
            return  # End turn after planning

        # --- State: EVACUATING ---
        if self.status == "EVACUATING":
            # Check if stuck in traffic and if patience has run out.
            if self.is_stuck_in_traffic():
                self.time_stuck_s += self.model.step_seconds
                if self.time_stuck_s > self.patience_threshold_s:
                    self.status = "PLANNING"  # Trigger replanning
                    self.time_stuck_s = 0
                    return
            else:
                self.time_stuck_s = 0  # Reset stuck timer if not stuck

            # Perform the movement.
            self.move()

    def plan_evacuation_route(self):
        """Determines agent's destination and calculates the path."""
        # Skip planning if agent is already failed
        if self.status == "FAILED":
            return

        # Destination Logic: Is home a safe option?
        target_node = None

        # First, try home if it's available and safe
        if self.home_location is not None:
            is_home_safe = not self.model.is_pos_in_evacuation_area(self.home_location)
            if is_home_safe:
                target_node = self.model.get_nearest_node(
                    self.home_location, self.main_mode
                )
                if target_node is not None:
                    print(f"Agent {self.original_id}: Going home (safe location)")
                    self.target_node = target_node
                else:
                    print(f"Agent {self.original_id}: Home is safe but unreachable")

        # If home isn't available or safe, find nearest shelter
        if target_node is None:
            print(
                f"Agent {self.original_id}: Seeking shelter (home unavailable or unsafe)"
            )
            target_node = self.model.get_nearest_shelter_node(
                self.current_pos_node, self.main_mode
            )

        self.target_node = target_node

        if self.target_node is None:
            print(f"Agent {self.original_id}: Could not find any target destination")
            self.status = "FAILED"
            return

        # Calculate the path using the model's pathfinding service.
        self.path = self.model.plan_route(self, self.current_pos_node, self.target_node)

        if self.path and len(self.path) >= 2:
            self.status = "EVACUATING"
            self.current_edge = None  # Reset edge state for new path
            print(
                f"Agent {self.original_id}: Route planned with {len(self.path)} nodes"
            )
        else:
            print(
                f"Agent {self.original_id}: Could not find path from {self.current_pos_node} to {self.target_node}"
            )
            self.status = "FAILED"  # Failed to find a path

    def move(self):
        """Moves the agent along its calculated path for one time step."""
        if not self.path or len(self.path) < 2:
            # This case can be triggered if path is empty or just the start node
            if self.current_pos_node == self.target_node:
                self.status = "ARRIVED"
            else:
                self.status = "FAILED"  # Stuck with no path
            return

        # Set current edge if we are starting a new one
        if self.current_edge is None:
            u, v = self.path[0], self.path[1]
            self.current_edge = (u, v)
            self.edge_progress_m = 0.0

        # Calculate distance to travel in this step
        distance_to_travel = self.speed_m_s * self.model.step_seconds

        # Traverse the path graph
        while distance_to_travel > 0:
            u, v = self.current_edge
            edge_data = self.model.get_graph_for_mode(self.main_mode).get_edge_data(
                u, v
            )

            # Handle case where edge doesn't exist
            if edge_data is None:
                print(f"Agent {self.original_id}: Edge ({u}, {v}) not found in graph")
                self.status = "FAILED"
                return

            # OSMnx graphs can have parallel edges, so we get the first one (key=0)
            edge_length = edge_data[0]["length"]

            remaining_on_edge = edge_length - self.edge_progress_m

            if distance_to_travel >= remaining_on_edge:
                # We will complete this edge and maybe move to the next
                self.current_pos_node = v
                self.path.pop(0)  # Advance path
                distance_to_travel -= remaining_on_edge

                # Report usage of the *completed* edge
                self.model.report_edge_usage(self, (u, v, 0))

                if not self.path or len(self.path) < 2:
                    # Arrived at the end of the path
                    self.status = "ARRIVED"
                    self.current_edge = None
                    break
                else:
                    # Set up the next edge
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

        congestion = self.model.get_edge_congestion(self.current_edge)
        return congestion > 1.0  # Stuck if congestion index is over 100%


# --- MODEL DEFINITION ---


class EvacuationModel(mesa.Model):
    """
    The main model class for the evacuation simulation.
    It holds the environment (graphs), manages agents, and collects data.
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

        # --- Environment Setup ---
        self.G_drive = G_drive
        self.G_walk = G_walk
        self.G_cycle = G_cycle
        self.evac_polygon = evacuation_area_polygon
        self.amenities_df = amenities_df

        # Pre-calculate amenity nodes for fast lookups
        print("Pre-calculating nearest nodes for amenities...")
        if len(amenities_df) > 0:
            try:
                # Convert to proper lists first
                lons = amenities_df["longitude"].to_list()
                lats = amenities_df["latitude"].to_list()

                amenity_nodes_result = ox.nearest_nodes(self.G_walk, X=lons, Y=lats)

                # Handle different return types from ox.nearest_nodes
                if hasattr(amenity_nodes_result, "__iter__") and not isinstance(
                    amenity_nodes_result, (str, int)
                ):
                    # It's an array or list-like object
                    import numpy as np

                    if isinstance(amenity_nodes_result, np.ndarray):
                        self.amenity_nodes = amenity_nodes_result.tolist()
                    else:
                        self.amenity_nodes = list(amenity_nodes_result)
                else:
                    # Single node case
                    self.amenity_nodes = [amenity_nodes_result]

                print(f"Found {len(self.amenity_nodes)} potential shelter nodes.")
            except Exception as e:
                print(f"Error pre-calculating amenity nodes: {e}")
                self.amenity_nodes = []
        else:
            print("No amenities provided!")
            self.amenity_nodes = []

        # --- Agent & Scheduling Setup ---
        # MESA 3.0+: No more schedulers! Agents are automatically managed by model.agents
        print(f"Creating {len(agents_df)} agents...")
        for agent_data in agents_df.iter_rows(named=True):
            agent_id = agent_data.get("ID")
            if agent_id is None:
                continue
            # MESA 3.0+: No unique_id parameter - automatically assigned
            agent = EvacuationAgent(
                model=self,
                original_id=agent_id,  # Store your original ID separately
                **dict(agent_data),
            )
            # MESA 3.0+: No need to add to schedule - agents are automatically added to model.agents

        print(f"Created {len(self.agents)} agents successfully.")

        # --- Data Collection & Bottleneck Monitoring ---
        self.edge_load = defaultdict(int)
        self.edge_agents = defaultdict(list)
        self.bottleneck_log = []

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

    def step(self):
        """Advance the model by one time step."""
        # 1. Reset counters for the new step
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

    def _analyze_and_log_bottlenecks(self):
        """Iterates through road usage and logs congested edges."""
        for edge, load in self.edge_load.items():
            try:
                # Capacity must be pre-calculated and added to your graph edges
                # A simple heuristic: capacity = number_of_lanes * vehicles_per_minute
                u, v, key = edge
                capacity = self.G_drive.edges[u, v, key].get(
                    "capacity", 20
                )  # Default capacity

                if load > capacity:
                    agents_on_edge = self.edge_agents[edge]
                    if agents_on_edge:  # Make sure we have agents
                        avg_svi = sum(a.svi for a in agents_on_edge) / len(
                            agents_on_edge
                        )
                        congestion_index = load / capacity

                        self.bottleneck_log.append(
                            {
                                "time": self.sim_time,
                                "edge_nodes": f"({u}, {v})",  # Convert to string for consistency
                                "load": load,
                                "capacity": capacity,
                                "congestion_index": congestion_index,
                                "avg_svi_stuck": avg_svi,
                            }
                        )
            except (KeyError, ZeroDivisionError):
                continue  # Edge may not be in drive graph or no agents on it

    # --- HELPER METHODS (API FOR AGENTS) ---

    def get_graph_for_mode(self, main_mode: str) -> nx.MultiDiGraph:
        """Returns the appropriate network graph based on the agent's mode."""
        if main_mode == "WALKING":
            return self.G_walk
        elif main_mode == "BIKE":
            return self.G_cycle
        else:  # CAR, PT, etc. all use the drive network
            return self.G_drive

    def get_nearest_node(self, pos: tuple, mode: str):
        """Finds the nearest node on the appropriate graph for a given lat/lon."""
        # Validate coordinates
        if pos is None or len(pos) != 2:
            print(f"Invalid position: {pos}")
            return None

        lat, lon = pos[0], pos[1]

        # Check if coordinates are valid numbers
        if lat is None or lon is None:
            print(f"None coordinates: lat={lat}, lon={lon}")
            return None

        try:
            # Convert to float if they're strings
            lat = float(lat)
            lon = float(lon)
        except (ValueError, TypeError):
            print(f"Non-numeric coordinates: lat={lat}, lon={lon}")
            return None

        # Check if coordinates are reasonable (basic sanity check)
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            # Try swapping if they seem to be in wrong order
            if -90 <= lon <= 90 and -180 <= lat <= 180:
                lat, lon = lon, lat
                print(f"Swapped coordinates: now lat={lat}, lon={lon}")
            else:
                print(f"Invalid coordinate range: lat={lat}, lon={lon}")
                return None

        graph = self.get_graph_for_mode(mode)
        try:
            return ox.distance.nearest_nodes(graph, X=lon, Y=lat)
        except Exception as e:
            print(f"Error finding nearest node for pos {pos}: {e}")
            return None

    def is_pos_in_evacuation_area(self, pos: tuple) -> bool:
        """Checks if a (lat, lon) point is inside the evacuation polygon."""
        if pos is None or len(pos) != 2:
            return False

        lat, lon = pos[0], pos[1]

        if lat is None or lon is None:
            return False

        try:
            lat = float(lat)
            lon = float(lon)
        except (ValueError, TypeError):
            return False

        return self.evac_polygon.contains(Point(lon, lat))

    def get_nearest_shelter_node(self, source_node: int, mode: str) -> int:
        """Finds the closest amenity (shelter) to an agent."""
        if source_node is None or not self.amenity_nodes:
            print(
                f"No shelter search possible: source_node={source_node}, amenity_nodes={len(self.amenity_nodes) if self.amenity_nodes else 0}"
            )
            return None

        graph = self.get_graph_for_mode(mode)

        try:
            # Find the nearest amenity to this source position
            min_distance = float("inf")
            nearest_shelter_node = None

            # Ensure amenity_nodes is a proper list of integers
            amenity_nodes_list = []
            if hasattr(self.amenity_nodes, "__iter__"):
                for node in self.amenity_nodes:
                    try:
                        # Convert to int if it's not already
                        if hasattr(node, "item"):  # numpy scalar
                            amenity_nodes_list.append(int(node.item()))
                        else:
                            amenity_nodes_list.append(int(node))
                    except (ValueError, TypeError):
                        continue

            if not amenity_nodes_list:
                print("No valid amenity nodes found")
                return None

            for amenity_node in amenity_nodes_list:
                try:
                    # Calculate distance from agent's current position to this amenity
                    distance = nx.shortest_path_length(
                        graph, source_node, amenity_node, weight="length"
                    )
                    if distance < min_distance:
                        min_distance = distance
                        nearest_shelter_node = amenity_node
                except nx.NetworkXNoPath:
                    # No path to this amenity, skip it
                    continue
                except Exception as e:
                    print(f"Error calculating path to amenity node {amenity_node}: {e}")
                    continue

            if nearest_shelter_node is not None:
                print(
                    f"Found shelter node {nearest_shelter_node} at distance {min_distance:.0f}m"
                )
            else:
                print("No reachable shelter found")

            return nearest_shelter_node

        except Exception as e:
            print(f"Error finding nearest shelter from node {source_node}: {e}")
            # Fallback: just return the first amenity node if everything fails
            if self.amenity_nodes:
                try:
                    fallback_node = self.amenity_nodes[0]
                    if hasattr(fallback_node, "item"):  # numpy scalar
                        fallback_node = int(fallback_node.item())
                    else:
                        fallback_node = int(fallback_node)
                    print(f"Using fallback shelter node: {fallback_node}")
                    return fallback_node
                except:
                    pass
            return None

    def plan_route(
        self, agent: EvacuationAgent, source_node: int, target_node: int
    ) -> list:
        """
        Calculates the shortest path for an agent, considering dynamic traffic.
        This is a core component of the simulation's intelligence.
        """
        graph = self.get_graph_for_mode(agent.main_mode)

        def dynamic_weight(u, v, attr):
            """A custom weight function for pathfinding."""
            # Base cost is travel time on an empty road
            length_m = attr.get("length", 0.0)
            base_travel_time = length_m / agent.speed_m_s

            # Penalty for traffic congestion from the *previous* time step
            congestion_index = self.get_edge_congestion(
                (u, v, 0)
            )  # Using key=0 for simplicity
            # An academic justification for this formula: We use an exponential penalty
            # to strongly discourage agents from entering already-jammed roads.
            # CI=1 means 2x cost, CI=2 means 4x cost.
            penalty_factor = max(1.0, 2**congestion_index)

            return base_travel_time * penalty_factor

        try:
            path = nx.shortest_path(
                graph, source_node, target_node, weight=dynamic_weight
            )
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
            print(f"No path found from {source_node} to {target_node}: {e}")
            return []  # Return empty list if no path exists

    def report_edge_usage(self, agent: EvacuationAgent, edge: tuple):
        """Called by agents to report which road they are on this step."""
        self.edge_load[edge] += 1
        self.edge_agents[edge].append(agent)

    def get_edge_congestion(self, edge: tuple) -> float:
        """Returns the congestion index for a given edge from the last step."""
        try:
            load = self.edge_load.get(edge, 0)
            u, v, key = edge
            capacity = self.G_drive.edges[u, v, key].get("capacity", 20)
            return load / capacity
        except (KeyError, ZeroDivisionError):
            return 0.0
