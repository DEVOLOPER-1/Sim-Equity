# FILE: evacuation_model_agentpy.py
# --------------------------------
# AgentPy implementation of the evacuation simulation
# Migrated from Mesa for better performance and maintainability

import time
import warnings
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import agentpy as ap
import networkx as nx
import osmnx as ox
import polars as pl
from shapely.geometry import Point

# Configure OSMnx for better performance
ox.settings.use_cache = True
ox.settings.log_console = False
ox.settings.timeout = 300


class EvacuationAgent(ap.Agent):
    """
    Represents a person from the NetMob25 dataset with their own unique
    vulnerability, assets, and behavioral parameters.
    """

    def setup(self, **kwargs):
        """Initialize agent properties with individual data."""
        # Store any individual agent data passed during setup
        if kwargs:
            self.agent_data = kwargs
            # Update self.p with individual agent data
            self.p.update(kwargs)
        else:
            self.agent_data = {}

        self.status = "INACTIVE"

        # Access agent-specific data (try both self.p and self.agent_data)
        self.svi = float(self._get_param("SVI_normalized", 0.0))
        self.main_mode = self._get_param("main_mode", "WALKING")

        # Location handling with better error checking
        self.home_location = self._get_location(
            "home_location_lat", "home_location_lon"
        )
        self.start_pos = self._get_location("start_lat", "start_lon")
        self.current_pos_node = None

        # Debug print to check what we're getting
        print(
            f"Agent {self.id}: start_pos = {self.start_pos}, home_location = {self.home_location}"
        )
        print(f"Agent {self.id}: Available params: {list(self.p.keys())}")

        # Check if we have valid location data
        if self.start_pos is None:
            self.status = "FAILED"
            self.fail_reason = "Missing location data"
            return  # Skip further initialization

        # Find the starting node on the graph
        self.current_pos_node = self.model.get_nearest_node(
            self.start_pos, self.main_mode
        )
        if self.current_pos_node is None:
            self.status = "FAILED"
            self.fail_reason = "Could not find valid starting node on graph"
            return  # Skip further initialization

        # Rest of the initialization...
        self.target_node = None
        self.path = []
        self.time_on_current_edge_s = 0.0
        self.replan_attempts = 0
        self.MAX_REPLAN_ATTEMPTS = 5
        self.fail_reason = None  # Track why agent failed

        # Time management and SVI-driven behavior
        self._init_behavioral_params()

    def _get_param(self, key, default=None):
        """Get parameter from agent data with fallback."""
        # First try individual agent data, then model parameters
        if hasattr(self, "agent_data") and key in self.agent_data:
            return self.agent_data[key]
        return self.p.get(key, default)

    def _get_location(
        self, lat_key: str, lon_key: str
    ) -> Optional[Tuple[float, float]]:
        """Safely extract and validate location coordinates from agent parameters."""
        lat = self._get_param(lat_key)
        lon = self._get_param(lon_key)

        try:
            if lat is not None and lon is not None:
                lat_float = float(lat)
                lon_float = float(lon)

                # Basic sanity check for coordinates
                if -90 <= lat_float <= 90 and -180 <= lon_float <= 180:
                    return (lat_float, lon_float)
                else:
                    print(f"Invalid coordinates: lat={lat_float}, lon={lon_float}")
                    return None
            else:
                print(f"Missing coordinates: lat={lat}, lon={lon}")
                return None
        except (ValueError, TypeError) as e:
            print(
                f"Error converting coordinates to float: lat={lat}, lon={lon}, error={e}"
            )
            return None

    def _init_behavioral_params(self):
        """Initialize all agent-specific timing and behavioral parameters based on SVI."""
        # 1. Reaction Delay
        start_time_str = self._get_param("start_time")
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
        base_speed_m_s = self._get_base_speed()
        self.speed_m_s = base_speed_m_s * (
            1.0 - self.svi * self.model.svi_speed_penalty
        )

        # 3. Patience/Rerouting
        self.patience_threshold_s = self.model.base_patience_s * (1.0 - self.svi)
        self.time_stuck_s = 0

    def _get_base_speed(self) -> float:
        """Get the appropriate base speed from the agent's data based on their main mode."""
        mode_speeds = {
            "WALKING": self._get_param("walking_speed_m_s", 1.4),
            "BIKE": self._get_param("cycling_speed_m_s", 4.5),
        }
        return float(
            mode_speeds.get(
                self.main_mode.upper(), self._get_param("median_speed_m_s", 8.3)
            )
        )

    def update(self):
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

        # Check if we've exceeded maximum replan attempts
        if self.replan_attempts >= self.MAX_REPLAN_ATTEMPTS:
            self.fail_agent("Exceeded maximum replanning attempts")
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
            self.fail_agent("Could not determine a valid destination")
            return

        # Calculate the path using the model's pathfinding service.
        self.path = self.model.plan_route(self, self.current_pos_node, self.target_node)

        if self.path:
            self.status = "EVACUATING"
            self.time_on_current_edge_s = 0.0  # Reset edge timer for the new path
            self.replan_attempts = 0  # Reset replan counter on success
        else:
            self.replan_attempts += 1
            warnings.warn(
                f"Agent {self.id}: Failed to find a valid path from {self.current_pos_node} to {self.target_node}."
            )
            # Don't set to FAILED yet, let it try again next step

    def move(self):
        """Moves the agent from one node to the next along its calculated path."""
        if not self.path or len(self.path) < 2:
            self.status = (
                "ARRIVED" if self.current_pos_node == self.target_node else "FAILED"
            )
            return

        u, v = self.path[0], self.path[1]

        # Get edge data
        graph = self.model.graphs[self.model._normalize_mode(self.main_mode)]
        edge_data = graph.edges[u, v] if (u, v) in graph.edges else None

        if edge_data is None:
            warnings.warn(
                f"Agent {self.id}: Edge ({u}, {v}) not found in graph. Replanning."
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

    def fail_agent(self, reason: str):
        """Helper method to fail an agent with a descriptive message"""
        warnings.warn(f"Agent {self.id}: {reason}. Setting status to FAILED.")
        self.status = "FAILED"


class EvacuationModel(ap.Model):
    def setup(self):
        """Initialize the model with parameters."""
        print("🚀 Initializing EvacuationModel...")
        init_start = time.time()

        # Simulation parameters (these stay in model.p)
        self.start_datetime = self.p.start_datetime
        self.sim_time = self.p.start_datetime
        self.step_seconds = self.p.step_seconds
        self.svi_speed_penalty = self.p.svi_speed_penalty
        self.max_svi_start_delay_s = self.p.max_svi_start_delay_s
        self.base_patience_s = self.p.base_patience_s

        # Environment data
        print("📊 Loading graphs...")
        self.graphs = {
            "CAR": ox.load_graphml(self.p.graphml_path_drive),
            "WALKING": ox.load_graphml(self.p.graphml_path_walk),
            "BIKE": ox.load_graphml(self.p.graphml_path_cycle),
        }
        print("✅ Graphs loaded")

        # Convert numeric attributes to floats
        for mode, graph in self.graphs.items():
            for node, data in graph.nodes(data=True):
                for attr in ["x", "y"]:
                    if attr in data and isinstance(data[attr], str):
                        try:
                            data[attr] = float(data[attr])
                        except ValueError:
                            pass

            for u, v, data in graph.edges(data=True):
                for attr in ["length", "capacity", "weight"]:
                    if attr in data and isinstance(data[attr], str):
                        try:
                            data[attr] = float(data[attr])
                        except ValueError:
                            pass

        # Environment data
        self.evac_polygon = self.p.evacuation_area_polygon

        # Pre-compute and cache the locations of safe shelters
        print("🏠 Pre-computing shelter nodes...")
        shelter_start = time.time()
        self.shelter_nodes = self._precompute_shelter_nodes(self.p.amenities_df)
        shelter_time = time.time() - shelter_start
        print(f"✅ Shelter nodes computed in {shelter_time:.2f}s")

        # Create agents - FIXED VERSION using AgentPy's built-in parameter passing
        print("👥 Creating agents...")
        agent_start = time.time()

        # Get the DataFrame from parameters
        agents_df: pl.DataFrame = self.p.agents_df

        # Convert to list of dictionaries for AgentPy
        agents_data = agents_df.to_dicts()

        print(f"   Processed {len(agents_data)} agent records")

        # Create agents using AgentPy's method but with individual parameters
        self.agents = ap.AgentList(self)

        for i, agent_data in enumerate(agents_data):
            # Create agent and pass individual data through setup
            agent = EvacuationAgent(self, agent_id=i)
            # Call setup again with individual data
            agent.setup(**agent_data)
            self.agents.append(agent)

        agent_time = time.time() - agent_start
        print(f"✅ Created {len(self.agents)} agents in {agent_time:.2f}s")

        # Bottleneck monitoring
        self.edge_load = defaultdict(int)
        self.edge_agents = defaultdict(list)
        self.bottleneck_log = []

        total_time = time.time() - init_start
        print(f"🎉 Model initialization complete in {total_time:.2f}s")

    def step(self):
        """Advance the model by one time step."""
        step_start = time.time()

        self.edge_load.clear()
        self.edge_agents.clear()
        self.bottleneck_log = []

        self.sim_time += timedelta(seconds=self.step_seconds)

        # Update all agents
        self.agents.update()

        # Analyze bottlenecks
        self._analyze_and_log_bottlenecks()

        step_time = time.time() - step_start

        # Count agent statuses for monitoring
        status_counts = defaultdict(int)
        for agent in self.agents:
            status_counts[agent.status] += 1

        print(
            f"⏰ Step {self.t}: {step_time:.3f}s | "
            f"Active: {status_counts['EVACUATING']}, "
            f"Planning: {status_counts['PLANNING']}, "
            f"Arrived: {status_counts['ARRIVED']}, "
            f"Failed: {status_counts['FAILED']}, "
            f"Inactive: {status_counts['INACTIVE']}"
        )

    def _precompute_shelter_nodes(self, amenities_df: pl.DataFrame) -> Dict[str, set]:
        """Finds nearest graph nodes for all out-of-bounds amenities."""
        shelters = {mode: set() for mode in self.graphs.keys()}
        if amenities_df.is_empty():
            print("   No amenities data provided")
            return shelters

        print(f"   Processing {len(amenities_df):,} amenities...")

        # Filter to safe amenities (outside evacuation zone)
        safe_amenities = []
        for row in amenities_df.iter_rows(named=True):
            if not self.is_pos_in_evacuation_area(
                (row["latitude"], row["longitude"]), False
            ):
                safe_amenities.append(row)

        print(f"   Filtered to {len(safe_amenities):,} safe amenities")

        if len(safe_amenities) == 0:
            return shelters

        # Process each mode
        for mode, graph in self.graphs.items():
            # Extract coordinates for batch processing
            amenity_coords = [
                (row["latitude"], row["longitude"]) for row in safe_amenities
            ]

            # Build KDTree for fast nearest node search
            node_points = []
            node_ids = []
            for node_id, data in graph.nodes(data=True):
                if "x" in data and "y" in data:
                    try:
                        x, y = float(data["x"]), float(data["y"])
                        node_points.append((y, x))  # (lat, lon)
                        node_ids.append(node_id)
                    except (ValueError, TypeError):
                        continue

            if not node_points:
                continue

            # Create KDTree
            from scipy.spatial import KDTree

            kdtree = KDTree(node_points)

            # Find nearest nodes for all amenities
            dists, idxs = kdtree.query(amenity_coords, k=1)
            nearest_nodes = [node_ids[i] for i in idxs]

            # Add to shelters set
            for node in nearest_nodes:
                if node is not None:
                    shelters[mode].add(node)

            print(f"   Found {len(nearest_nodes):,} shelter nodes for {mode}")

        total_shelters = sum(len(s) for s in shelters.values())
        print(f"   Found {total_shelters:,} unique shelter locations across all modes")

        return shelters

    def _analyze_and_log_bottlenecks(self):
        """Iterates through road usage and logs congested edges."""
        for edge, load in self.edge_load.items():
            u, v = edge
            try:
                # Use CAR graph for edge data
                graph = self.graphs["CAR"]
                edge_data = graph.edges[u, v]
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

    def get_nearest_node(self, pos: tuple, mode: str) -> Optional[int]:
        """Finds the nearest node in the graph to the given position."""
        mode = self._normalize_mode(mode)
        graph = self.graphs.get(mode)
        if not graph or pos is None:
            return None

        # Build KDTree for fast nearest node search
        node_points = []
        node_ids = []
        for node_id, data in graph.nodes(data=True):
            if "x" in data and "y" in data:
                try:
                    x, y = float(data["x"]), float(data["y"])
                    node_points.append((y, x))  # (lat, lon)
                    node_ids.append(node_id)
                except (ValueError, TypeError):
                    continue

        if not node_points:
            return None

        # Create KDTree and find nearest node
        from scipy.spatial import KDTree

        kdtree = KDTree(node_points)
        dist, idx = kdtree.query([pos], k=1)
        return node_ids[idx[0]]

    def get_nearest_shelter_node(self, source_node: int, mode: str) -> Optional[int]:
        """Finds the closest pre-computed shelter to an agent's current node."""
        mode = self._normalize_mode(mode)
        shelters = self.shelter_nodes.get(mode)
        if not shelters or source_node is None:
            return None

        graph = self.graphs.get(mode)
        if not graph:
            return None

        # Check if source node is already a shelter
        if source_node in shelters:
            return source_node

        # Compute distance to each shelter on-demand
        try:
            min_distance = float("inf")
            nearest_shelter = None

            for shelter in shelters:
                try:
                    path_length = nx.shortest_path_length(
                        graph, source=source_node, target=shelter, weight="length"
                    )
                    if path_length < min_distance:
                        min_distance = path_length
                        nearest_shelter = shelter
                except nx.NetworkXNoPath:
                    continue

            return nearest_shelter

        except Exception as e:
            print(f"Error finding nearest shelter for {mode}: {e}")
            return None

    def plan_route(
        self, agent: EvacuationAgent, source_node: int, target_node: int
    ) -> list:
        """Calculates the shortest path using dynamic weight function, considering traffic."""
        mode = self._normalize_mode(agent.main_mode)
        graph = self.graphs.get(mode)
        if not graph or source_node is None or target_node is None:
            return []

        # Check if source and target are the same
        if source_node == target_node:
            return [source_node]

        agent_speed = max(agent.speed_m_s, 0.1)

        def dynamic_weight(u, v, data):
            """Dynamic weight function that considers congestion for car agents"""
            base_length = data.get("length", 1.0)

            if mode == "CAR":
                # Convert to travel time
                base_travel_time = base_length / agent_speed

                # Get congestion for this edge
                congestion = self.get_edge_congestion((u, v))
                penalty_factor = 2**congestion  # Exponential penalty

                return base_travel_time * penalty_factor
            else:
                # For non-car modes, use length directly
                return base_length

        try:
            return nx.shortest_path(
                graph, source=source_node, target=target_node, weight=dynamic_weight
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
            graph = self.graphs["CAR"]
            edge_data = graph.edges[u, v]
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
        return mode_mapping.get(mode, "CAR")


def run_simulation(parameters):
    """Run the evacuation simulation with the given parameters."""
    # Create model
    model = EvacuationModel(parameters)

    # Run simulation
    results = model.run(steps=parameters.get("steps", 60), display=True)

    return model, results
