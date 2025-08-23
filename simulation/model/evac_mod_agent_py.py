# FILE: evacuation_model_agentpy.py
# --------------------------------
# AgentPy implementation of the evacuation simulation
# Migrated from Mesa for better performance and maintainability
import os
import time
import warnings
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

import agentpy as ap
import networkx as nx
import osmnx as ox
import polars as pl
from haversine import haversine
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
        self.unique_id = str(self._get_param("ID", ""))
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

        # Initialize path history FIRST
        self.path_history: list[Dict[str, Any]] = []

        # Add starting position to path history - SIMPLIFIED AND FIXED
        self._add_position_to_history(step=0, force_add=True)

        # Rest of the initialization...
        self.target_node = None
        self.path = []
        self.time_on_current_edge_s = 0.0
        self.replan_attempts = 0
        self.MAX_REPLAN_ATTEMPTS = 5
        self.fail_reason = None  # Track why agent failed

        # Time management and SVI-driven behavior
        self._init_behavioral_params()

    def _add_position_to_history(self, step=None, force_add=False):
        """Helper method to add current position to path history with proper error handling."""
        try:
            # Use current model step if not provided
            if step is None:
                step = getattr(self.model, "t", 0)

            # Get current simulation time
            current_time = getattr(self.model, "sim_time", self.model.start_datetime)
            if isinstance(current_time, datetime):
                current_time = current_time.isoformat()

            # Validate we have necessary data
            if not self.current_pos_node:
                if not force_add:  # Only warn if not forced (like initial setup)
                    print(f"Agent {self.id}: No current position node to log")
                return False

            if not self.main_mode:
                print(f"Agent {self.id}: No main mode defined")
                return False

            # Get the appropriate graph and node data
            normalized_mode = self.model._normalize_mode(self.main_mode)
            if normalized_mode not in self.model.graphs:
                print(
                    f"Agent {self.id}: Mode {normalized_mode} not in available graphs"
                )
                return False

            graph = self.model.graphs[normalized_mode]
            if self.current_pos_node not in graph:
                print(
                    f"Agent {self.id}: Node {self.current_pos_node} not in {normalized_mode} graph"
                )
                return False

            node_data = graph.nodes[self.current_pos_node]
            if "x" not in node_data or "y" not in node_data:
                print(
                    f"Agent {self.id}: Node {self.current_pos_node} missing coordinates"
                )
                return False

            # Create the history entry
            history_entry = {
                "step": step,
                "time": current_time,
                "x": float(node_data["x"]),
                "y": float(node_data["y"]),
                "mode": self.main_mode,
                "status": self.status,
            }

            # Add to path history
            self.path_history.append(history_entry)

            # Debug output for first few entries
            if len(self.path_history) <= 3:
                print(
                    f"Agent {self.id}: Added path history entry {len(self.path_history)}: {history_entry}"
                )

            return True

        except Exception as e:
            print(f"Agent {self.id}: Error adding position to history: {e}")
            import traceback

            traceback.print_exc()
            return False

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
                # Log the status change
                self._add_position_to_history()
            else:
                return

        self.evacuation_time += self.model.step_seconds

        if self.status == "PLANNING":
            self.plan_evacuation_route()
            # Log after planning
            if self.status == "EVACUATING":  # Planning was successful
                self._add_position_to_history()

        if self.status == "EVACUATING":
            if self.is_stuck_in_traffic():
                self.time_stuck_s += self.model.step_seconds
                if self.time_stuck_s > self.patience_threshold_s:
                    self.status = "PLANNING"  # Trigger replanning
                    self.time_stuck_s = 0
                    self._add_position_to_history()  # Log replanning
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
            self.home_location, True
        )
        if is_home_safe:
            home_node = self.model.get_nearest_node(self.home_location, self.main_mode)
            # ADDED: Verify home node is actually outside evacuation area
            if home_node and not self.model.is_node_in_evacuation_area(
                home_node, self.main_mode
            ):
                self.target_node = home_node

        # If home is not an option, find the nearest designated shelter.
        if self.target_node is None:
            self.target_node = self.model.get_nearest_shelter_node(
                self, source_node=self.current_pos_node, mode=self.main_mode
            )

        if self.target_node is None:
            self.fail_agent("Could not determine a valid destination")
            return

        # ADDED: Final verification that target node is outside evacuation area
        if self.model.is_node_in_evacuation_area(self.target_node, self.main_mode):
            self.fail_agent("Target node is inside evacuation area")
            return

        # Calculate the path using the model's pathfinding service.
        self.path = self.model.plan_route_astar(
            self, self.current_pos_node, self.target_node
        )

        if self.path:
            self.status = "EVACUATING"
            self.time_on_current_edge_s = 0.0  # Reset edge timer for the new path
            self.replan_attempts = 0  # Reset replan counter on success
            print(
                f"Agent {self.id}: Successfully planned route with {len(self.path)} nodes"
            )
        else:
            self.replan_attempts += 1
            warnings.warn(
                f"Agent {self.id}: Failed to find a valid path from {self.current_pos_node} to {self.target_node}."
            )

    def move(self):
        """Moves the agent from one node to the next along its calculated path."""
        if not self.path or len(self.path) < 2:
            self.status = (
                "ARRIVED" if self.current_pos_node == self.target_node else "FAILED"
            )
            # Log final status
            self._add_position_to_history()
            return

        u, v = self.path[0], self.path[1]

        # Get edge data - properly handling multigraphs
        graph = self.model.graphs[self.model._normalize_mode(self.main_mode)]
        edge_data = None

        try:
            if nx.is_multigraphical(graph):
                # For multigraphs, we need an edge key
                if graph.has_edge(u, v):
                    # Get the first key (or a specific key if needed)
                    keys = list(graph[u][v].keys())
                    if keys:
                        edge_data = graph[u][v][keys[0]]
            else:
                # For simple graphs
                if graph.has_edge(u, v):
                    edge_data = graph[u][v]
        except Exception as e:
            warnings.warn(f"Error accessing edge data: {e}")
            edge_data = None

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
            old_node = self.current_pos_node
            self.current_pos_node = v
            self.path.pop(0)  # Advance to the next node in the path
            self.time_on_current_edge_s = 0.0

            print(
                f"Agent {self.id}: Moved from node {old_node} to {self.current_pos_node}"
            )

            # ALWAYS log position after moving to a new node
            self._add_position_to_history()

        else:
            # Agent continues along the current edge
            self.time_on_current_edge_s += self.model.step_seconds
            # Even if not moving to a new node, we can log intermediate positions
            # But only occasionally to avoid too much data
            if self.model.t % 5 == 0:  # Every 5 steps
                self._add_position_to_history()

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
        self.fail_reason = reason
        # Log the failure
        self._add_position_to_history()


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

        # Initialize agent paths DataFrame
        self.agent_paths_df = pl.DataFrame(
            schema={
                "agent_id": pl.Int32,
                "svi": pl.Float32,
                "main_mode": pl.Utf8,
                "status": pl.Utf8,
                "evacuation_time": pl.Int32,
                "start_lat": pl.Float64,
                "start_lon": pl.Float64,
                "end_lat": pl.Float64,
                "end_lon": pl.Float64,
                "fail_reason": pl.Utf8,
                "started_at": pl.Utf8,
                "arrived_at": pl.Utf8,
            }
        )

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

        # Debug: Check path history for a few agents every 10 steps
        if self.t % 10 == 0:
            agents_with_history = [
                a
                for a in self.agents
                if hasattr(a, "path_history") and len(a.path_history) > 0
            ]
            print(
                f"Debug: {len(agents_with_history)} agents have path history at step {self.t}"
            )
            if agents_with_history:
                sample_agent = agents_with_history[0]
                print(
                    f"Sample agent {sample_agent.id} has {len(sample_agent.path_history)} history entries"
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
            # Amenities CSV has latitude, longitude columns
            lat, lon = row["latitude"], row["longitude"]
            if not self.is_pos_in_evacuation_area((lat, lon), if_lat_lon=True):
                safe_amenities.append(row)

        print(f"   Filtered to {len(safe_amenities):,} safe amenities")

        if len(safe_amenities) == 0:
            return shelters

        # Process each mode
        for mode, graph in self.graphs.items():
            # Extract coordinates for batch processing - amenities are (lat, lon)
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
                        # OSMnx: x=lon, y=lat, convert to (lat, lon) for KDTree
                        node_points.append((y, x))  # (lat, lon)
                        node_ids.append(node_id)
                    except (ValueError, TypeError):
                        continue

            if not node_points:
                continue

            # Create KDTree - both amenity_coords and node_points are (lat, lon)
            from scipy.spatial import KDTree

            kdtree = KDTree(node_points)

            # Find nearest nodes for all amenities
            dists, idxs = kdtree.query(amenity_coords, k=1)
            nearest_nodes = [node_ids[i] for i in idxs]

            # ADDED: Verify nodes are outside evacuation area
            for i, node_id in enumerate(nearest_nodes):
                if node_id is not None:
                    node_data = graph.nodes[node_id]
                    # Get node coordinates (x=lon, y=lat)
                    node_lat, node_lon = node_data["y"], node_data["x"]
                    # Check if node is outside evacuation area
                    if not self.is_pos_in_evacuation_area(
                        (node_lat, node_lon), if_lat_lon=True
                    ):
                        shelters[mode].add(node_id)
                    else:
                        print(
                            f"   Excluding node {node_id} as it's inside evacuation area"
                        )

            print(f"   Found {len(shelters[mode]):,} shelter nodes for {mode}")

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
                edge_data = None

                if nx.is_multigraphical(graph):
                    if graph.has_edge(u, v):
                        keys = list(graph[u][v].keys())
                        if keys:
                            edge_data = graph[u][v][keys[0]]
                else:
                    if graph.has_edge(u, v):
                        edge_data = graph[u][v]

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
            except (KeyError, TypeError, ZeroDivisionError) as e:
                continue

    # --- HELPER METHODS (API FOR AGENTS) ---

    def is_pos_in_evacuation_area(self, pos: tuple, if_lat_lon: bool = True) -> bool:
        """Checks if a point is inside the evacuation polygon.

        Args:
            pos: Position as either (lat, lon) or (lon, lat)
            if_lat_lon: If True, pos is (lat, lon); if False, pos is (lon, lat)

        Returns:
            True if point is inside evacuation area
        """
        if if_lat_lon:
            # Convert (lat, lon) to (lon, lat) for Shapely
            lon, lat = pos[1], pos[0]
        else:
            # Already in (lon, lat) format
            lon, lat = pos[0], pos[1]

        return self.evac_polygon.contains(Point(lon, lat))

    def is_node_in_evacuation_area(self, node_id: int, mode: str) -> bool:
        """Check if a graph node is inside the evacuation area."""
        mode = self._normalize_mode(mode)
        graph = self.graphs.get(mode)
        if not graph or node_id not in graph.nodes:
            return False

        node_data = graph.nodes[node_id]
        # OSMnx: x=lon, y=lat
        node_lat, node_lon = node_data.get("y"), node_data.get("x")

        if node_lat is None or node_lon is None:
            return False

        return self.is_pos_in_evacuation_area((node_lat, node_lon), if_lat_lon=True)

    def get_nearest_node(self, pos: tuple, mode: str) -> Optional[int]:
        """Finds the nearest node in the graph to the given position.

        Args:
            pos: Position as (lat, lon) tuple
            mode: Transportation mode

        Returns:
            Nearest node ID or None
        """
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
                    # OSMnx stores x=longitude, y=latitude
                    # Convert to (lat, lon) for KDTree to match input format
                    node_points.append((y, x))  # (lat, lon)
                    node_ids.append(node_id)
                except (ValueError, TypeError):
                    continue

        if not node_points:
            return None

        # Create KDTree and find nearest node
        from scipy.spatial import KDTree

        kdtree = KDTree(node_points)
        dist, idx = kdtree.query([pos], k=1)  # pos is (lat, lon)
        return node_ids[idx[0]]

    def get_nearest_shelter_node(
        self, agent: EvacuationAgent, source_node: int, mode: str
    ) -> Optional[int]:
        """Finds the closest shelter using single-source Dijkstra."""
        shelter_start = time.time()

        mode = self._normalize_mode(mode)
        shelters = self.shelter_nodes.get(mode)
        if not shelters or source_node is None:
            shelter_time = time.time() - shelter_start
            print(
                f"No shelters available for Agent {agent.id} ({mode}) in {shelter_time:.3f}s "
                f"(searched at {self.sim_time.strftime('%H:%M:%S')})"
            )
            return None

        graph = self.graphs.get(mode)
        if not graph:
            shelter_time = time.time() - shelter_start
            print(
                f"No graph available for Agent {agent.id} ({mode}) in {shelter_time:.3f}s "
                f"(searched at {self.sim_time.strftime('%H:%M:%S')})"
            )
            return None

        if source_node in shelters:
            shelter_time = time.time() - shelter_start
            print(
                f"Agent {agent.id} already at shelter in {shelter_time:.3f}s "
                f"(found at {self.sim_time.strftime('%H:%M:%S')})"
            )
            return source_node

        try:
            # Single Dijkstra run - stops when all shelters are found
            distances = nx.single_source_dijkstra_path_length(
                graph, source_node, weight="length", cutoff=None
            )

            # Find nearest shelter from computed distances
            min_distance = float("inf")
            nearest_shelter = None

            for shelter in shelters:
                if shelter in distances and distances[shelter] < min_distance:
                    min_distance = distances[shelter]
                    nearest_shelter = shelter

            shelter_time = time.time() - shelter_start

            if nearest_shelter and self.is_node_in_evacuation_area(
                nearest_shelter, mode
            ):
                print(
                    f"Warning: Shelter node {nearest_shelter} is inside evacuation area"
                )
                return None

            if nearest_shelter:
                print(
                    f"Nearest shelter found for Agent {agent.id} in {shelter_time:.3f}s "
                    f"(distance: {min_distance:.0f}m, searched at {self.sim_time.strftime('%H:%M:%S')})"
                )
            else:
                print(
                    f"No reachable shelter found for Agent {agent.id} in {shelter_time:.3f}s "
                    f"(searched at {self.sim_time.strftime('%H:%M:%S')})"
                )

            return nearest_shelter

        except nx.NetworkXError:
            shelter_time = time.time() - shelter_start
            print(
                f"Shelter search failed for Agent {agent.id} (NetworkX error) in {shelter_time:.3f}s "
                f"(at {self.sim_time.strftime('%H:%M:%S')})"
            )
            return None

        except Exception as e:
            shelter_time = time.time() - shelter_start
            print(
                f"Shelter search error for Agent {agent.id} in {shelter_time:.3f}s: {e} "
                f"(at {self.sim_time.strftime('%H:%M:%S')})"
            )
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

    def plan_route_astar(
        self, agent: EvacuationAgent, source_node: int, target_node: int
    ) -> list:
        """
        Calculates the shortest path using A* algorithm with geographic heuristic.
        Much faster than Dijkstra for large geographical networks.
        """
        astar_start = time.time()

        mode = self._normalize_mode(agent.main_mode)
        graph = self.graphs.get(mode)
        if not graph or source_node is None or target_node is None:
            return []

        # Check if source and target are the same
        if source_node == target_node:
            return [source_node]

        agent_speed = max(agent.speed_m_s, 0.1)

        # Get target node coordinates for heuristic calculation
        target_data = graph.nodes.get(target_node)
        if not target_data or "x" not in target_data or "y" not in target_data:
            # Fallback to Dijkstra if we can't get coordinates
            return self.plan_route(agent, source_node, target_node)

        target_x, target_y = float(target_data["x"]), float(target_data["y"])

        def heuristic(node_id: int, _target=None) -> float:
            """
            Geographic heuristic: Euclidean distance to target.
            This is admissible (never overestimates) for geographical networks.

            Note: _target is ignored as we already have target_node in closure
            """
            node_data = graph.nodes.get(node_id)
            if not node_data or "x" not in node_data or "y" not in node_data:
                return 0.0

            node_x, node_y = float(node_data["x"]), float(node_data["y"])

            # Calculate haversine distance with correct lat/lon format
            # OSMnx stores x=longitude, y=latitude
            node_coords = (node_y, node_x)  # (lat, lon)
            target_coords = (target_y, target_x)  # (lat, lon)

            haversine_dist = haversine(node_coords, target_coords, unit="m")

            # Convert to travel time estimate
            if mode == "CAR":
                return haversine_dist / agent_speed * 1.1  # 10% buffer
            else:
                return haversine_dist

        def dynamic_weight(u, v, data):
            """Dynamic weight function that considers congestion for car agents"""
            base_length = data.get("length", 1.0)

            if mode == "CAR":
                base_travel_time = base_length / agent_speed
                congestion = self.get_edge_congestion((u, v))
                penalty_factor = 2**congestion
                return base_travel_time * penalty_factor
            else:
                return base_length

        try:
            # Use A* with proper weight function (not string)
            path = nx.astar_path(
                graph,
                source=source_node,
                target=target_node,
                heuristic=heuristic,
                weight=dynamic_weight,
            )

            astar_time = time.time() - astar_start
            print(
                f"A* path found for Agent {agent.id} in {astar_time:.3f}s "
                f"(planned at {self.sim_time.strftime('%H:%M:%S')})"
            )

            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound, KeyError) as e:
            astar_time = time.time() - astar_start
            print(f"A* failed for Agent {agent.id} after {astar_time:.3f}s: {e}")
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
            edge_data = None

            if nx.is_multigraphical(graph):
                if graph.has_edge(u, v):
                    keys = list(graph[u][v].keys())
                    if keys:
                        edge_data = graph[u][v][keys[0]]
            else:
                if graph.has_edge(u, v):
                    edge_data = graph[u][v]

            capacity = float(edge_data.get("capacity", 5))
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

    def collect_agent_paths_data(self) -> pl.DataFrame:
        """Collect path history data from all agents into the Polars DataFrame."""
        print("📊 Collecting agent path history data...")

        # Create lists for each column
        agent_ids = []
        svi_values = []
        main_modes = []
        statuses = []
        evac_times = []
        start_lats = []
        start_lons = []
        end_lats = []
        end_lons = []
        fail_reasons = []
        started_ats = []
        arrived_ats = []

        # Ensure the traces directory exists
        traces_dir = "simulation_outcomes/agents_traces"
        os.makedirs(traces_dir, exist_ok=True)

        # Debug: Count agents with path history
        agents_with_history = sum(
            1
            for agent in self.agents
            if hasattr(agent, "path_history") and agent.path_history
        )
        print(
            f"Debug: {agents_with_history} out of {len(self.agents)} agents have path history"
        )

        # Collect data from all agents
        for agent in self.agents:
            # Get start and end positions from path history if available
            start_lat, start_lon, end_lat, end_lon = None, None, None, None
            started_at, arrived_at = None, None

            if hasattr(agent, "path_history") and agent.path_history:
                print(
                    f"Agent {agent.id}: Has {len(agent.path_history)} path history entries"
                )

                # Write trace CSV for this agent
                if len(agent.path_history) >= 1:
                    try:
                        his_df = pl.DataFrame(agent.path_history)
                        trace_file = f"{traces_dir}/{agent.unique_id}.csv"
                        his_df.write_csv(trace_file)
                        print(
                            f"✅ Written trace for agent {agent.unique_id} ({len(agent.path_history)} entries)"
                        )
                    except Exception as e:
                        print(
                            f"❌ Error writing trace for agent {agent.unique_id}: {e}"
                        )
                        import traceback

                        traceback.print_exc()

                if len(agent.path_history) > 0:
                    first_entry = agent.path_history[0]
                    start_lat = first_entry.get("y")
                    start_lon = first_entry.get("x")
                    started_at = first_entry.get("time")
                    if isinstance(started_at, datetime):
                        started_at = started_at.isoformat()

                if len(agent.path_history) > 0:
                    last_entry = agent.path_history[-1]
                    end_lat = last_entry.get("y")
                    end_lon = last_entry.get("x")
                    arrived_at = last_entry.get("time")
                    if isinstance(arrived_at, datetime):
                        arrived_at = arrived_at.isoformat()
            else:
                print(f"Agent {agent.id}: No path history available")

            # Append data to lists
            agent_ids.append(agent.unique_id)
            svi_values.append(getattr(agent, "svi", None))
            main_modes.append(getattr(agent, "main_mode", None))
            statuses.append(getattr(agent, "status", None))
            evac_times.append(getattr(agent, "evacuation_time", 0))
            start_lats.append(start_lat)
            start_lons.append(start_lon)
            end_lats.append(end_lat)
            end_lons.append(end_lon)
            fail_reasons.append(getattr(agent, "fail_reason", None))
            started_ats.append(started_at)
            arrived_ats.append(arrived_at)

        # Create DataFrame from collected data
        self.agent_paths_df = pl.DataFrame(
            data={
                "agent_id": agent_ids,
                "svi": svi_values,
                "main_mode": main_modes,
                "status": statuses,
                "evacuation_time": evac_times,
                "start_lat": start_lats,
                "start_lon": start_lons,
                "end_lat": end_lats,
                "end_lon": end_lons,
                "fail_reason": fail_reasons,
                "started_at": started_ats,
                "arrived_at": arrived_ats,
            }
        )

        print(f"✅ Collected path data for {len(agent_ids)} agents")
        return self.agent_paths_df


def run_simulation(parameters):
    """Run the evacuation simulation with the given parameters."""
    # Create output directories
    os.makedirs("simulation_outcomes", exist_ok=True)
    os.makedirs("simulation_outcomes/agents_traces", exist_ok=True)

    # Create model
    model = EvacuationModel(parameters)

    # Run simulation
    results = model.run(steps=parameters.get("steps", 60), display=True)

    # Collect agent paths data after simulation
    agent_paths_df = model.collect_agent_paths_data()

    # Add the DataFrame to the results for later use in analytics
    results.agent_paths_df = agent_paths_df

    # Write the DataFrame to CSV
    agent_paths_df.write_csv("simulation_outcomes/Agents_Statistics_Trial.csv")
    print("✅ Successfully wrote agent statistics to CSV")

    return model, results
