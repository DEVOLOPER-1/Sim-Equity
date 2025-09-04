# FILE: evacuation_model_agentpy_enhanced.py
# --------------------------------
# Enhanced AgentPy implementation with multi-modal transportation
# Integrated R5py for public transport routing in Île-de-France
#
# Key Features:
# - Multi-modal routing: WALKING (fallback), BIKE, CAR, PUBLIC_TRANSPORT
# - Seamless mode switching (bike-to-transit, walk-to-transit)
# - Robust fallback mechanisms
# - Comprehensive agent tracking and data collection
# --------------------------------

import os
import time
import traceback
import warnings
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

import agentpy as ap
import geopandas
import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd
import polars as pl
import r5py
from haversine import haversine
from shapely import wkt
from shapely.geometry import Point

# Configure OSMnx for better performance
ox.settings.use_cache = True
ox.settings.log_console = False
ox.settings.timeout = 300


class EvacuationAgent(ap.Agent):
    """
    Represents a person with multi-modal evacuation capabilities.

    Transport Modes:
    - WALKING: Primary fallback mode, always available
    - BIKE: Can switch to walking or public transport
    - CAR: Road-based routing with congestion consideration
    - PUBLIC_TRANSPORT: R5py-based multi-modal routing

    Attributes:
        unique_id (str): Unique identifier for the agent
        svi (float): Social Vulnerability Index (0.0 to 1.0)
        main_mode (str): Primary transportation mode
        status (str): Current status (INACTIVE, PLANNING, EVACUATING, ARRIVED, FAILED)
        current_pos_node (int): Current position on the network graph
        path (List[int]): Current routing path (OSMnx nodes)
        journey_plan (List[Dict]): Multi-modal journey segments (R5py)
        evacuation_time (float): Total evacuation time in seconds
    """

    def ensure_required_attributes(self) -> None:
        """
        Ensure all required attributes exist to prevent attribute errors.
        This method is called during setup to initialize all agent properties.
        """
        required_attrs = {
            # Multi-modal transport attributes
            "using_public_transport": False,
            "journey_plan": [],
            "current_journey_segment": 0,
            "nearest_transit_stop": None,
            "original_mode": None,  # Store original mode before fallback
            # Agent data and status
            "agent_data": {},
            "path_history": [],
            "status": "INACTIVE",
            "fail_reason": None,
            # Movement and timing
            "replan_attempts": 0,
            "time_on_current_edge_s": 0.0,
            "time_stuck_s": 0.0,
            "evacuation_time": 0.0,
            # Navigation
            "path": [],
            "target_node": None,
            "current_pos_node": None,
        }

        for attr, default in required_attrs.items():
            if not hasattr(self, attr):
                setattr(self, attr, default)

    def setup(self, **kwargs) -> None:
        """
        Initialize agent properties with individual data.

        Args:
            **kwargs: Agent-specific data including location, mode, SVI, etc.
        """
        # Ensure all required attributes exist first
        self.ensure_required_attributes()

        # Store agent data
        self.agent_data = kwargs if kwargs else {}
        if kwargs:
            self.p.update(kwargs)

        # Basic agent properties
        self.status = "INACTIVE"
        self.unique_id = str(self._get_param("ID", f"agent_{self.id}"))
        self.svi = float(self._get_param("SVI_normalized", 0.0))

        # Transport mode setup with fallback chain
        requested_mode = self._get_param("main_mode", "WALKING")
        self.original_mode = requested_mode
        self.main_mode = self._setup_transport_mode(requested_mode)

        # Location handling
        self.home_location = self._get_location(
            "home_location_lat", "home_location_lon"
        )
        self.start_pos = self._get_location("start_lat", "start_lon")

        # Validate location data
        if self.start_pos is None:
            self._fail_agent("Missing or invalid location data")
            return

        # Find starting node with mode fallback
        self.current_pos_node = self._find_starting_node()
        if self.current_pos_node is None:
            self._fail_agent(
                "Could not find valid starting node on any available graph"
            )
            return

        # Initialize path tracking
        self.path_history: List[Dict[str, Any]] = []
        self._add_position_to_history(step=0, force_add=True)

        # Initialize behavioral parameters based on SVI
        self._init_behavioral_params()

        print(
            f"Agent {self.unique_id}: Initialized with mode {self.main_mode} at node {self.current_pos_node}"
        )

    def _setup_transport_mode(self, requested_mode: str) -> str:
        """
        Setup transport mode with fallback chain.

        Args:
            requested_mode (str): The requested transportation mode

        Returns:
            str: The final assigned mode (may be different due to fallbacks)
        """
        normalized_mode = self.model._normalize_mode(requested_mode)

        # Check if requested mode is available
        if normalized_mode in self.model.graphs:
            return normalized_mode

        # Fallback chain: requested -> WALKING (always available)
        print(
            f"Agent {self.unique_id}: Mode {requested_mode} not available, falling back to WALKING"
        )

        if "WALKING" not in self.model.graphs:
            raise ValueError("WALKING graph is required but not available")

        return "WALKING"

    def _find_starting_node(self) -> Optional[int]:
        """
        Find starting node with mode fallback.

        Returns:
            Optional[int]: Starting node ID or None if no valid node found
        """
        # Try with current mode first
        node = self.model.get_nearest_node(self.start_pos, self.main_mode)
        if node is not None:
            return node

        # Fall back to walking mode
        if self.main_mode != "WALKING":
            print(
                f"Agent {self.unique_id}: Could not find node for {self.main_mode}, trying WALKING"
            )
            self.main_mode = "WALKING"
            node = self.model.get_nearest_node(self.start_pos, "WALKING")
            if node is not None:
                return node

        return None

    def _get_param(self, key: str, default: Any = None) -> Any:
        """
        Get parameter from agent data with fallback to defaults.

        Args:
            key (str): Parameter name
            default (Any): Default value if parameter not found

        Returns:
            Any: Parameter value or default
        """
        if hasattr(self, "agent_data") and key in self.agent_data:
            return self.agent_data[key]
        return self.p.get(key, default)

    def _get_location(
        self, lat_key: str, lon_key: str
    ) -> Optional[Tuple[float, float]]:
        """
        Safely extract and validate location coordinates.

        Args:
            lat_key (str): Key for latitude value
            lon_key (str): Key for longitude value

        Returns:
            Optional[Tuple[float, float]]: (lat, lon) tuple or None if invalid
        """
        lat = self._get_param(lat_key)
        lon = self._get_param(lon_key)

        try:
            if lat is not None and lon is not None:
                lat_float = float(lat)
                lon_float = float(lon)
                if -90 <= lat_float <= 90 and -180 <= lon_float <= 180:
                    return (lat_float, lon_float)
        except (ValueError, TypeError):
            pass
        return None

    def _init_behavioral_params(self) -> None:
        """
        Initialize agent-specific timing and behavioral parameters based on SVI.
        Higher SVI leads to delayed activation, slower movement, and less patience.
        """
        # Reaction Delay (SVI affects when agent starts evacuating)
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

        # Speed Penalty (SVI reduces movement speed)
        base_speed_m_s = self._get_base_speed()
        self.speed_m_s = base_speed_m_s * (
            1.0 - self.svi * self.model.svi_speed_penalty
        )

        # Patience Threshold (SVI reduces patience for traffic)
        self.patience_threshold_s = self.model.base_patience_s * (1.0 - self.svi)

        # Reset timing counters
        self.evacuation_time = 0.0
        self.time_stuck_s = 0.0

    def _get_base_speed(self) -> float:
        """
        Get appropriate base speed based on current transport mode.

        Returns:
            float: Base speed in meters per second
        """
        mode_speeds = {
            "WALKING": self._get_param("walking_speed_m_s", 1.4),  # ~5 km/h
            "BIKE": self._get_param("cycling_speed_m_s", 4.5),  # ~16 km/h
            "CAR": self._get_param("driving_speed_m_s", 13.9),  # ~50 km/h
        }
        return mode_speeds.get(self.main_mode, self._get_param("median_speed_m_s", 1.4))

    def update(self) -> None:
        """
        Execute agent's logic for a single simulation time step.
        Main state machine: INACTIVE -> PLANNING -> EVACUATING -> ARRIVED/FAILED
        """
        if self.status in ["ARRIVED", "FAILED"]:
            return

        # Check if agent should activate
        if self.status == "INACTIVE":
            if self.model.sim_time >= self.effective_activation_time:
                self.status = "PLANNING"
                print(f"Agent {self.unique_id}: Activated for evacuation")
                self._add_position_to_history()
            else:
                return

        # Update evacuation timer
        self.evacuation_time += self.model.step_seconds

        # Plan route if needed
        if self.status == "PLANNING":
            self.plan_evacuation_route()
            if self.status == "EVACUATING":
                self._add_position_to_history()

        # Move if evacuating
        if self.status == "EVACUATING":
            # Handle traffic congestion (cars only)
            if self.is_stuck_in_traffic():
                self.time_stuck_s += self.model.step_seconds
                if self.time_stuck_s > self.patience_threshold_s:
                    print(f"Agent {self.unique_id}: Stuck in traffic, replanning route")
                    self.status = "PLANNING"
                    self.time_stuck_s = 0.0
                    self._add_position_to_history()
                    return
            else:
                self.time_stuck_s = 0.0

            # Execute movement
            self.move()

    def plan_evacuation_route(self) -> None:
        """
        Determine destination and calculate evacuation route.
        Supports both single-mode and multi-modal routing.
        """
        if self.status == "FAILED":
            return

        if self.replan_attempts >= self.model.MAX_REPLAN_ATTEMPTS:
            self._fail_agent(
                f"Exceeded maximum replanning attempts ({self.model.MAX_REPLAN_ATTEMPTS})"
            )
            return

        # Determine safe destination
        self.target_node = self._determine_destination()
        if self.target_node is None:
            self.replan_attempts += 1
            return

        # Validate destination is safe
        if self.model.is_node_in_evacuation_area(self.target_node, self.main_mode):
            self._fail_agent("Target destination is inside evacuation area")
            return

        # Choose routing strategy based on mode and availability
        success = False

        if self.main_mode != "CAR" and self.model.use_public_transport:
            # Try multi-modal routing for bike/walk + transit
            success = self._plan_multi_modal_route()

        if not success:
            # Fall back to single-mode routing
            success = self._plan_single_mode_route()

        if success:
            self.status = "EVACUATING"
            self.time_on_current_edge_s = 0.0
            self.replan_attempts = 0
            print(f"Agent {self.unique_id}: Route planned successfully")
        else:
            self.replan_attempts += 1
            warnings.warn(
                f"Agent {self.unique_id}: Failed to plan route (attempt {self.replan_attempts})"
            )

    def _determine_destination(self) -> Optional[int]:
        """
        Determine safe evacuation destination.
        Priority: 1) Safe home location, 2) Nearest shelter

        Returns:
            Optional[int]: Target node ID or None if no valid destination
        """
        # Check if home is safe and reachable
        if self.home_location and self.home_location != self.start_pos:
            is_home_safe = not self.model.is_pos_in_evacuation_area(
                self.home_location, True
            )
            if is_home_safe:
                home_node = self.model.get_nearest_node(
                    self.home_location, self.main_mode
                )
                if home_node and not self.model.is_node_in_evacuation_area(
                    home_node, self.main_mode
                ):
                    print(f"Agent {self.unique_id}: Heading home")
                    return home_node

        # Find nearest shelter
        shelter_node = self.model.get_nearest_shelter_node(
            self, source_node=self.current_pos_node, mode=self.main_mode
        )

        if shelter_node:
            print(f"Agent {self.unique_id}: Heading to shelter")
            return shelter_node

        # If no shelter found, try to find any safe location outside evacuation zone
        safe_node = self._find_any_safe_node()
        if safe_node:
            print(f"Agent {self.unique_id}: Heading to safe location")
            return safe_node

        self._fail_agent("Could not find any safe destination")
        return None

    def _find_any_safe_node(self) -> Optional[int]:
        """
        Find any safe node outside evacuation area as fallback.
        """
        graph = self._get_movement_graph()
        if not graph:
            return None

        # Try to find a node in the general direction away from evacuation area
        for node_id, data in graph.nodes(data=True):
            if "x" in data and "y" in data:
                node_lat, node_lon = data["y"], data["x"]
                if not self.model.is_pos_in_evacuation_area((node_lat, node_lon), True):
                    # Check if this node is reachable
                    try:
                        path = nx.shortest_path(
                            graph, self.current_pos_node, node_id, weight="length"
                        )
                        if path:
                            return node_id
                    except nx.NetworkXNoPath:
                        continue

        return None

    def _plan_multi_modal_route(self) -> bool:
        try:
            print(f"Agent {self.unique_id}: Attempting multi-modal routing")

            # Find nearest transit stop
            self.nearest_transit_stop = self.model.find_nearest_transit_stop(
                self.start_pos
            )
            if not self.nearest_transit_stop:
                print(f"Agent {self.unique_id}: No nearby transit stop found")
                return False

            dest_coords = self.model.get_node_coordinates(self.target_node, "WALKING")
            if not dest_coords:
                print(f"Agent {self.unique_id}: Could not get destination coordinates")
                return False

            # Plan journey using R5py
            journey_df = self.model.plan_public_transport_journey(
                self.start_pos, dest_coords, self.model.sim_time
            )

            # Defensive diagnostics
            if journey_df is None:
                print(
                    f"Agent {self.unique_id}: plan_public_transport_journey returned None"
                )
                return False

            print(
                f"Agent {self.unique_id}: journey type: {type(journey_df)}; columns: {list(getattr(journey_df, 'columns', []))}"
            )

            # Correct check for empty DataFrame (do NOT call .is_empty())
            if getattr(journey_df, "empty", False):
                print(f"Agent {self.unique_id}: journey_df is empty (no itineraries)")
                # Optionally show a sample of the object for debugging:
                try:
                    print(
                        "journey_df preview:",
                        getattr(journey_df, "head", lambda: None)(),
                    )
                except Exception:
                    pass
                return False

            # If you specifically want to check geometry emptiness (optional)
            if "geometry" in getattr(journey_df, "columns", []):
                try:
                    if journey_df.geometry.is_empty.all():
                        print(f"Agent {self.unique_id}: all geometry values are empty")
                        return False
                except Exception as e:
                    print(
                        f"Agent {self.unique_id}: geometry emptiness check failed: {e}"
                    )

            # Process journey into segments
            self.journey_plan = self.model.process_r5py_journey(journey_df)
            if not self.journey_plan:
                print(f"Agent {self.unique_id}: Failed to process journey plan")
                return False

            # Set up for multi-modal movement
            self.using_public_transport = True
            self.current_journey_segment = 0

            first_segment = self.journey_plan[0]
            if first_segment.get("mode", "").upper() in ["WALK", "WALKING"]:
                transit_node = self.model.get_nearest_node(
                    self.nearest_transit_stop, "WALKING"
                )
                if transit_node:
                    self.path = self.model.plan_route_astar(
                        self, self.current_pos_node, transit_node
                    )

            print(
                f"Agent {self.unique_id}: Multi-modal route planned with {len(self.journey_plan)} segments"
            )
            return True

        except Exception as e:
            import traceback

            print(f"Agent {self.unique_id}: Multi-modal planning error: {e}")
            traceback.print_exc()
            return False

    def _plan_single_mode_route(self) -> bool:
        """
        Plan single-mode route using OSMnx graphs.

        Returns:
            bool: True if successful, False if failed
        """
        try:
            self.path = self.model.plan_route_astar(
                self, self.current_pos_node, self.target_node
            )
            self.using_public_transport = False
            return len(self.path) > 0

        except Exception as e:
            print(f"Agent {self.unique_id}: Single-mode planning error: {e}")
            return False

    def move(self) -> None:
        """
        Move agent along calculated path or journey plan.
        Handles both single-mode and multi-modal movement.
        """
        if self.using_public_transport and self.journey_plan:
            self._move_multi_modal()
        else:
            self._move_single_mode()

    def _move_single_mode(self) -> None:
        """
        Standard movement along OSMnx graph path.
        Handles edge traversal with timing and congestion.
        """
        if not self.path or len(self.path) < 2:
            # Check if we've reached destination
            if self.current_pos_node == self.target_node:
                self.status = "ARRIVED"
                print(f"Agent {self.unique_id}: Arrived at destination")
            else:
                self._fail_agent("No valid path and not at destination")
            self._add_position_to_history()
            return

        # Get current edge to traverse
        u, v = self.path[0], self.path[1]
        graph = self._get_movement_graph()
        if graph is None:
            self._fail_agent("No valid graph available for movement")
            return

        edge_data = self.model.get_edge_data(u, v, graph)
        if edge_data is None:
            warnings.warn(
                f"Agent {self.unique_id}: Edge ({u}, {v}) not found. Replanning."
            )
            self.status = "PLANNING"
            return

        # Calculate edge traversal time
        edge_length_m = edge_data.get("length", 1.0)
        agent_speed = max(self.speed_m_s, 0.1)  # Prevent division by zero
        time_to_traverse_edge_s = edge_length_m / agent_speed

        # Report edge usage for traffic monitoring
        self.model.report_edge_usage(self, (u, v))

        # Check if edge will be completed this step
        if (
            self.time_on_current_edge_s + self.model.step_seconds
            >= time_to_traverse_edge_s
        ):
            # Complete edge traversal
            self.current_pos_node = v
            self.path.pop(0)
            self.time_on_current_edge_s = 0.0
            self._add_position_to_history()
        else:
            # Continue along edge
            self.time_on_current_edge_s += self.model.step_seconds
            # Update position occasionally for tracking
            if self.model.t % 5 == 0:
                self._add_position_to_history()

    def _move_multi_modal(self) -> None:
        """
        Movement along R5py multi-modal journey.
        FIXED: Proper time progression, destination tracking, and status updates
        """
        if self.current_journey_segment >= len(self.journey_plan):
            # CRITICAL FIX: Update current_pos_node to target destination
            if hasattr(self, "target_node") and self.target_node:
                self.current_pos_node = self.target_node

            self.status = "ARRIVED"
            print(f"Agent {self.unique_id}: Completed multi-modal journey")
            self._add_position_to_history()
            return

        segment = self.journey_plan[self.current_journey_segment]
        segment_mode = segment.get("mode", "").upper()

        # FIXED: Convert duration from minutes to seconds for consistency
        segment_duration_minutes = segment.get("duration", 0)
        segment_duration_seconds = segment_duration_minutes * 60  # Convert to seconds

        print(
            f"Agent {self.unique_id}: Segment {self.current_journey_segment + 1}/{len(self.journey_plan)} - {segment_mode} ({segment_duration_minutes:.1f} min)"
        )

        # Handle waiting for transit
        if segment_mode in ["WAIT", "TRANSIT_WAIT"]:
            self.time_on_current_edge_s += self.model.step_seconds

            # Check if waiting is complete
            if self.time_on_current_edge_s >= segment_duration_seconds:
                self.current_journey_segment += 1
                self.time_on_current_edge_s = 0.0
                print(
                    f"Agent {self.unique_id}: Finished waiting ({segment_duration_minutes:.1f} min)"
                )

            self._add_position_to_history()
            return

        # Handle moving segments (walking, transit, etc.)
        self.time_on_current_edge_s += self.model.step_seconds

        # Calculate progress for this segment
        progress = (
            self.time_on_current_edge_s / segment_duration_seconds
            if segment_duration_seconds > 0
            else 1.0
        )

        # FIXED: Proper segment completion check
        if self.time_on_current_edge_s >= segment_duration_seconds:
            self.current_journey_segment += 1
            self.time_on_current_edge_s = 0.0

            print(
                f"Agent {self.unique_id}: Completed segment {segment_mode} ({segment_duration_minutes:.1f} min)"
            )

            if self.current_journey_segment >= len(self.journey_plan):
                # CRITICAL FIX: Update position to destination before marking as arrived
                if hasattr(self, "target_node") and self.target_node:
                    self.current_pos_node = self.target_node

                self.status = "ARRIVED"
                print(f"Agent {self.unique_id}: Multi-modal journey complete - ARRIVED")
            else:
                # Move to next segment
                next_segment = self.journey_plan[self.current_journey_segment]
                next_mode = next_segment.get("mode", "UNKNOWN")
                next_duration = next_segment.get("duration", 0)
                print(
                    f"Agent {self.unique_id}: Starting segment {self.current_journey_segment + 1}: {next_mode} ({next_duration:.1f} min)"
                )

        # FIXED: Always update position for tracking, especially near journey end
        self._add_position_to_history()

    def _get_movement_graph(self) -> Optional[nx.Graph]:
        """
        Get appropriate graph for movement based on current mode.

        Returns:
            Optional[nx.Graph]: Graph for movement or None if not available
        """
        graph = self.model.graphs.get(self.main_mode)

        # Fall back to walking graph if current mode not available
        if graph is None:
            graph = self.model.graphs.get("WALKING")
            if graph is None:
                return None

        return graph

    def is_stuck_in_traffic(self) -> bool:
        """
        Check if agent is stuck in traffic congestion.
        Only applicable to car agents on congested edges.

        Returns:
            bool: True if stuck in traffic, False otherwise
        """
        if self.main_mode != "CAR" or not self.path or len(self.path) < 2:
            return False

        next_edge = (self.path[0], self.path[1])
        return self.model.get_edge_congestion(next_edge) > 1.0

    def _add_position_to_history(
        self, step: Optional[int] = None, force_add: bool = False
    ) -> bool:
        """
        Add current position to movement history for tracking and analysis.

        Args:
            step (Optional[int]): Simulation step number
            force_add (bool): Force addition even if position unchanged

        Returns:
            bool: True if position was added, False otherwise
        """
        try:
            if step is None:
                step = getattr(self.model, "t", 0)

            current_time = getattr(self.model, "sim_time", self.model.start_datetime)
            if isinstance(current_time, datetime):
                current_time = current_time.isoformat()

            # Get current coordinates
            x, y = self._get_current_coordinates()
            if x is None or y is None and not force_add:
                return False

            # Create history entry
            history_entry = {
                "step": step,
                "time": current_time,
                "x": float(x) if x is not None else None,
                "y": float(y) if y is not None else None,
                "mode": getattr(self, "main_mode", "UNKNOWN"),
                "status": getattr(self, "status", "UNKNOWN"),
                "using_public_transport": getattr(
                    self, "using_public_transport", False
                ),
                "current_segment": getattr(self, "current_journey_segment", 0),
                "evacuation_time": getattr(self, "evacuation_time", 0.0),
            }

            self.path_history.append(history_entry)
            return True

        except Exception as e:
            print(f"Agent {self.unique_id}: Error adding position to history: {e}")
            return False

    def _get_current_coordinates(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Get current coordinates based on movement mode and position.
        FIXED: Proper interpolation and destination tracking for multi-modal journeys
        """
        # Multi-modal movement coordinate interpolation
        if (
            self.using_public_transport
            and hasattr(self, "journey_plan")
            and self.journey_plan
            and self.current_journey_segment < len(self.journey_plan)
        ):
            segment = self.journey_plan[self.current_journey_segment]

            # CRITICAL FIX: Check if this is the final segment
            is_final_segment = (
                self.current_journey_segment == len(self.journey_plan) - 1
            )

            progress = (
                self.time_on_current_edge_s / segment.get("duration", 1)
                if segment.get("duration", 0) > 0
                else 0
            )
            progress = min(max(progress, 0.0), 1.0)  # Clamp to [0, 1]

            # Try to interpolate along geometry if available
            if "geometry" in segment and segment["geometry"] is not None:
                try:
                    if (
                        is_final_segment and progress >= 0.95
                    ):  # Near end of final segment
                        # Return the destination coordinates instead of interpolated position
                        geom = segment["geometry"]
                        if hasattr(geom, "coords"):
                            # Get the last coordinate (destination)
                            coords = list(geom.coords)
                            if coords:
                                return coords[-1][1], coords[-1][0]  # (lat, lon)

                    point = segment["geometry"].interpolate(progress, normalized=True)
                    return point.y, point.x  # Return (lat, lon) not (x, y)
                except Exception:
                    pass

            # FALLBACK: If no geometry, try to get destination coordinates for final segment
            if is_final_segment and hasattr(self, "target_node") and self.target_node:
                dest_coords = self.model.get_node_coordinates(
                    self.target_node, "WALKING"
                )
                if dest_coords:
                    return dest_coords  # Already in (lat, lon) format

        # Standard node-based positioning
        if self.current_pos_node is not None:
            graph = self._get_movement_graph()
            if graph and self.current_pos_node in graph.nodes:
                node_data = graph.nodes[self.current_pos_node]
                # OSMnx: x=lon, y=lat -> return as (lat, lon)
                return node_data.get("y"), node_data.get("x")

        return None, None

    def _fail_agent(self, reason: str) -> None:
        """
        Mark agent as failed with descriptive reason.

        Args:
            reason (str): Reason for failure
        """
        print(f"Agent {self.unique_id}: FAILED - {reason}")
        self.status = "FAILED"
        self.fail_reason = reason
        self._add_position_to_history()


class EvacuationModel(ap.Model):
    """
    Enhanced evacuation simulation model with multi-modal transport support.

    Features:
    - Multiple transportation graphs (walk, bike, car)
    - R5py integration for public transport
    - Dynamic traffic congestion modeling
    - Social Vulnerability Index (SVI) integration
    - Comprehensive data collection and analysis
    """

    # Model constants
    MAX_REPLAN_ATTEMPTS: int = 5

    def setup(self) -> None:
        """Initialize the evacuation model with all components."""
        print("🚀 Initializing Enhanced EvacuationModel...")
        init_start = time.time()

        # Initialize simulation parameters
        self._setup_simulation_parameters()

        # Initialize data structures
        self._initialize_data_structures()

        # Load transportation graphs
        self._load_transportation_graphs()

        # Setup environment and amenities
        self._setup_environment()

        # Initialize public transport network
        self._setup_public_transport()

        # Create agents
        self._create_agents()

        # Initialize monitoring systems
        self._setup_monitoring()

        total_time = time.time() - init_start
        print(f"🎉 Model initialization complete in {total_time:.2f}s")

    def _setup_simulation_parameters(self) -> None:
        """Initialize core simulation parameters."""
        self.start_datetime: datetime = self.p.start_datetime
        self.sim_time: datetime = self.p.start_datetime.replace(year=2025, month=9)
        self.step_seconds = self.p.step_seconds
        base_step_seconds = self.p.step_seconds
        # SVI-based behavioral parameters
        self.svi_speed_penalty = self.p.get("svi_speed_penalty", 0.3)
        self.max_svi_start_delay_s = self.p.get("max_svi_start_delay_s", 1800)  # 30 min
        self.base_patience_s = self.p.get("base_patience_s", 300)  # 5 min

        # Public transport settings
        self.use_public_transport = self.p.get("use_public_transport", False)

        if self.use_public_transport:
            # Minimum step size to handle 1-minute transit segments
            min_step_for_transit = 60  # 1 minute in seconds
            if base_step_seconds > min_step_for_transit:
                print(
                    f"WARNING: Step size ({base_step_seconds}s) may be too large for transit simulation"
                )
                print(f"Consider reducing to {min_step_for_transit}s or smaller")

            # Recommended: 30-60 seconds for good transit simulation granularity
            recommended_step = min(base_step_seconds, 60)
            if recommended_step != base_step_seconds:
                print(
                    f"Adjusting step size from {base_step_seconds}s to {recommended_step}s for transit compatibility"
                )
                self.step_seconds = recommended_step
            else:
                self.step_seconds = base_step_seconds
        else:
            self.step_seconds = base_step_seconds

    def _initialize_data_structures(self) -> None:
        """Initialize data collection structures."""
        self.agent_paths_df = pl.DataFrame(
            schema={
                "agent_id": pl.Utf8,
                "svi": pl.Float32,
                "original_mode": pl.Utf8,
                "final_mode": pl.Utf8,
                "status": pl.Utf8,
                "evacuation_time": pl.Float32,
                "start_lat": pl.Float64,
                "start_lon": pl.Float64,
                "end_lat": pl.Float64,
                "end_lon": pl.Float64,
                "fail_reason": pl.Utf8,
                "started_at": pl.Utf8,
                "arrived_at": pl.Utf8,
                "used_public_transport": pl.Boolean,
                "segments_completed": pl.Int32,
            }
        )

    def _load_transportation_graphs(self) -> None:
        """Load and validate transportation network graphs."""
        print("Loading transportation graphs...")
        self.graphs = {}

        graph_paths = {
            "CAR": self.p.get("graphml_path_drive"),
            "WALKING": self.p.get("graphml_path_walk"),
            "BIKE": self.p.get("graphml_path_cycle"),
        }

        for mode, path in graph_paths.items():
            if path and os.path.exists(path):
                try:
                    print(f"Loading {mode} graph from {path}...")
                    graph = ox.load_graphml(path)
                    self.graphs[mode] = graph
                    print(
                        f"✓ {mode} graph loaded: {len(graph.nodes())} nodes, {len(graph.edges())} edges"
                    )
                except Exception as e:
                    print(f"✗ Failed to load {mode} graph from {path}: {e}")
                    if mode == "WALKING":
                        # Walking graph is mandatory
                        raise ValueError(
                            f"WALKING graph is required but failed to load: {e}"
                        )
            else:
                print(f"⚠ Graph path for {mode} not provided or doesn't exist: {path}")

        # Ensure we have at least a walking graph
        if "WALKING" not in self.graphs:
            raise ValueError("WALKING graph is required but not available")

        # Process and validate graph attributes
        self._process_graph_attributes()
        print("✓ All graphs loaded and processed")

    def _process_graph_attributes(self) -> None:
        """Process and validate graph node/edge attributes."""
        for mode, graph in self.graphs.items():
            # Convert string coordinates to float
            for node, data in graph.nodes(data=True):
                for attr in ["x", "y"]:
                    if attr in data and isinstance(data[attr], str):
                        try:
                            data[attr] = float(data[attr])
                        except ValueError:
                            print(
                                f"Warning: Invalid {attr} coordinate for node {node} in {mode} graph"
                            )

            # Convert string edge attributes to float
            for u, v, data in graph.edges(data=True):
                for attr in ["length", "capacity", "weight"]:
                    if attr in data and isinstance(data[attr], str):
                        try:
                            data[attr] = float(data[attr])
                        except ValueError:
                            if attr == "length":
                                data[attr] = 1.0  # Default length
                            elif attr == "capacity":
                                data[attr] = 20.0  # Default capacity

    def _setup_environment(self) -> None:
        """Setup evacuation area and shelter locations."""
        print("Setting up environment...")

        # Load evacuation area polygon
        self.evac_polygon = self.p.evacuation_area_polygon

        # Pre-compute shelter nodes for all transport modes
        print("Pre-computing shelter nodes...")
        shelter_start = time.time()
        self.shelter_nodes = self._precompute_shelter_nodes(self.p.amenities_df)
        shelter_time = time.time() - shelter_start
        print(f"✓ Shelter nodes computed in {shelter_time:.2f}s")

    def _setup_public_transport(self) -> None:
        """Initialize R5py transport network for public transport routing."""
        if not self.use_public_transport:
            print("Public transport disabled")
            return

        print("Initializing R5py transport network...")
        try:
            osm_pbf_path = self.p.get("osm_pbf_path", "ile-de-france-latest.osm.pbf")
            gtfs_zip_path = self.p.get("gtfs_zip_path", "idfm-gtfs.zip")

            if os.path.exists(osm_pbf_path) and os.path.exists(gtfs_zip_path):
                self.transport_network = r5py.TransportNetwork(
                    osm_pbf=osm_pbf_path,
                    gtfs=[gtfs_zip_path],  # R5py expects a list
                )
                print("✓ R5py transport network initialized")
            else:
                print(
                    f"✗ R5py files not found: OSM={osm_pbf_path}, GTFS={gtfs_zip_path}"
                )
                self.use_public_transport = False
        except Exception as e:
            print(f"✗ Failed to initialize R5py: {e}")
            self.use_public_transport = False

    def _create_agents(self) -> None:
        """Create and initialize evacuation agents."""
        print("Creating agents...")
        agent_start = time.time()

        agents_df: pl.DataFrame = self.p.agents_df
        agents_data = agents_df.to_dicts()

        self.agents = ap.AgentList(self)
        for i, agent_data in enumerate(agents_data):
            try:
                agent = EvacuationAgent(self, agent_id=i)
                agent.setup(**agent_data)
                self.agents.append(agent)
            except Exception as e:
                print(f"Warning: Failed to create agent {i}: {e}")

        agent_time = time.time() - agent_start
        print(f"✓ Created {len(self.agents)} agents in {agent_time:.2f}s")

    def _setup_monitoring(self) -> None:
        """Initialize monitoring and data collection systems."""
        # Traffic congestion monitoring
        self.edge_load = defaultdict(int)
        self.edge_agents = defaultdict(list)
        self.bottleneck_log = []

        # Transit stops cache for performance
        self.transit_stops_cache = {}

    def step(self) -> None:
        """Advance the model by one time step."""
        step_start = time.time()

        # Clear monitoring data from previous step
        self.edge_load.clear()
        self.edge_agents.clear()
        self.bottleneck_log = []

        # Update simulation time
        self.sim_time += timedelta(seconds=self.step_seconds)

        # Update all agents
        self.agents.update()

        # Analyze traffic bottlenecks
        self._analyze_bottlenecks()

        # Log progress
        self._log_step_progress(time.time() - step_start)

    def _analyze_bottlenecks(self) -> None:
        """Analyze and log traffic congestion bottlenecks."""
        for edge, load in self.edge_load.items():
            u, v = edge
            try:
                graph = self.graphs.get("CAR")
                if not graph:
                    continue

                edge_data = self.get_edge_data(u, v, graph)
                if not edge_data:
                    continue

                capacity = edge_data.get("capacity", 20)
                if load > capacity:
                    # Calculate average SVI of stuck agents
                    stuck_agents = self.edge_agents.get(edge, [])
                    avg_svi = (
                        sum(a.svi for a in stuck_agents) / len(stuck_agents)
                        if stuck_agents
                        else 0
                    )

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

    def _log_step_progress(self, step_time: float) -> None:
        """Log simulation step progress and agent status counts."""
        status_counts = defaultdict(int)
        mode_counts = defaultdict(int)

        for agent in self.agents:
            status_counts[agent.status] += 1
            mode_counts[getattr(agent, "main_mode", "UNKNOWN")] += 1

        print(
            f"Step {self.t}: {step_time:.3f}s | "
            f"Active: {status_counts['EVACUATING']}, "
            f"Planning: {status_counts['PLANNING']}, "
            f"Arrived: {status_counts['ARRIVED']}, "
            f"Failed: {status_counts['FAILED']}, "
            f"Inactive: {status_counts['INACTIVE']}"
        )

    def _precompute_shelter_nodes(
        self, amenities_df: pl.DataFrame
    ) -> Dict[str, Set[int]]:
        """
        Find nearest graph nodes for all safe amenities (outside evacuation zone).

        Args:
            amenities_df (pl.DataFrame): DataFrame containing amenity locations

        Returns:
            Dict[str, Set[int]]: Shelter nodes by transport mode
        """
        shelters = {mode: set() for mode in self.graphs.keys()}

        if amenities_df.is_empty():
            print("No amenities data provided")
            return shelters

        print(f"Processing {len(amenities_df):,} amenities...")

        # Filter to safe amenities (outside evacuation zone)
        safe_amenities = []
        for row in amenities_df.iter_rows(named=True):
            lat, lon = row["latitude"], row["longitude"]
            if not self.is_pos_in_evacuation_area((lat, lon), if_lat_lon=True):
                safe_amenities.append(row)

        print(f"Filtered to {len(safe_amenities):,} safe amenities")
        if not safe_amenities:
            return shelters

        # Process each transport mode
        for mode, graph in self.graphs.items():
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

            # Find nearest nodes using KDTree
            from scipy.spatial import KDTree

            kdtree = KDTree(node_points)
            dists, idxs = kdtree.query(amenity_coords, k=1)

            # Verify nodes are outside evacuation area
            for i, idx in enumerate(idxs):
                node_id = node_ids[idx]
                node_data = graph.nodes[node_id]
                node_lat, node_lon = node_data["y"], node_data["x"]

                if not self.is_pos_in_evacuation_area(
                    (node_lat, node_lon), if_lat_lon=True
                ):
                    shelters[mode].add(node_id)

            print(f"Found {len(shelters[mode]):,} shelter nodes for {mode}")

        total_shelters = sum(len(s) for s in shelters.values())
        print(f"Found {total_shelters:,} unique shelter locations across all modes")
        return shelters

    def find_nearest_transit_stop(
        self, position: Tuple[float, float]
    ) -> Optional[Tuple[float, float]]:
        """
        Find nearest public transport stop using R5py network analysis.

        Args:
            position (Tuple[float, float]): Starting position as (lat, lon)

        Returns:
            Optional[Tuple[float, float]]: Nearest stop coordinates or None
        """
        position_key = f"{position[0]:.6f}_{position[1]:.6f}"
        if position_key in self.transit_stops_cache:
            return self.transit_stops_cache[position_key]

        if not hasattr(self, "transport_network") or self.transport_network is None:
            print("Transport network not available.")
            return None

        try:
            # Create origins DataFrame from raw dictionary
            origins_df = geopandas.pd.DataFrame(
                {"id": [0], "lat": [position[0]], "lon": [position[1]]}
            )

            # *** FIX 1: Create and set the geometry for origins ***
            origins = geopandas.GeoDataFrame(
                origins_df,
                geometry=geopandas.points_from_xy(origins_df.lon, origins_df.lat),
                crs="EPSG:4326",
            )

            # Create a grid of nearby destinations to test transit connectivity
            search_radius = 0.015  # ~1.5km in degrees
            destinations_data = []
            dest_id = 0

            for lat_offset in [-search_radius, 0, search_radius]:
                for lon_offset in [-search_radius, 0, search_radius]:
                    destinations_data.append(
                        {
                            "id": dest_id,
                            "lat": position[0] + lat_offset,
                            "lon": position[1] + lon_offset,
                        }
                    )
                    dest_id += 1

            destinations_df = geopandas.pd.DataFrame(destinations_data)

            # *** FIX 2: Create and set the geometry for destinations ***
            destinations = geopandas.GeoDataFrame(
                destinations_df,
                geometry=geopandas.points_from_xy(
                    destinations_df.lon, destinations_df.lat
                ),
                crs="EPSG:4326",
            )

            # Use TravelTimeMatrix to find transit-accessible points
            # *** FIX 3: The TravelTimeMatrix class now computes automatically on creation ***
            matrix = r5py.TravelTimeMatrix(
                transport_network=self.transport_network,
                origins=origins,
                destinations=destinations,
                transport_modes=[r5py.TransportMode.TRANSIT],
                departure=self.sim_time,
                max_time=timedelta(minutes=30),
            )

            if matrix is not None and not matrix.empty:
                # Find the destination with shortest travel time
                shortest_time_row = matrix.sort_values("travel_time").iloc[0]
                dest_id = int(shortest_time_row["to_id"])  # ensure integer index

                # Get coordinates of best destination (represents transit stop area)
                best_dest = destinations_data[dest_id]
                stop_coords = (best_dest["lat"], best_dest["lon"])

                self.transit_stops_cache[position_key] = stop_coords
                return stop_coords

        except Exception as e:
            # Added a more detailed error log to help with future debugging
            import traceback

            print(f"Error finding transit stop: {e}")
            traceback.print_exc()

        return None

    def plan_public_transport_journey(
        self,
        origin: Tuple[float, float],
        destination: Tuple[float, float],
        departure_time: datetime,
    ) -> Optional[gpd.GeoDataFrame]:
        if not hasattr(self, "transport_network") or self.transport_network is None:
            return None

        try:
            # build pandas DataFrame directly (avoid mixing polars here)
            origins_df = pd.DataFrame(
                {"id": [0], "lat": [origin[0]], "lon": [origin[1]]}
            )
            destinations_df = pd.DataFrame(
                {"id": [0], "lat": [destination[0]], "lon": [destination[1]]}
            )

            # debug: ensure columns exist
            # print("origins columns:", origins_df.columns)
            # print("destinations columns:", destinations_df.columns)

            origins = gpd.GeoDataFrame(
                origins_df,
                geometry=gpd.points_from_xy(origins_df["lon"], origins_df["lat"]),
                crs="EPSG:4326",
            )

            destinations = gpd.GeoDataFrame(
                destinations_df,
                geometry=gpd.points_from_xy(
                    destinations_df["lon"], destinations_df["lat"]
                ),
                crs="EPSG:4326",
            )

            detailed_itineraries = r5py.DetailedItineraries(
                transport_network=self.transport_network,
                origins=origins,
                destinations=destinations,
                departure=departure_time,
                transport_modes=[r5py.TransportMode.TRANSIT, r5py.TransportMode.WALK],
                max_time=timedelta(hours=1),
                # max_rides=3,
                access_modes=[r5py.TransportMode.WALK],
                egress_modes=[r5py.TransportMode.WALK],
            )

            # DetailedItineraries returns a GeoDataFrame-like object
            # journey = detailed_itineraries
            return detailed_itineraries

        except Exception as e:
            print(f"Error planning public transport journey: {e}")
            # optionally log full traceback for debugging
            import traceback

            traceback.print_exc()
            return None

    def process_r5py_journey(self, journey_df: Any) -> List[Dict[str, Any]]:
        """
        Convert R5py journey GeoDataFrame to internal segment format.

        Expected journey_df columns (typical r5py output):
            ["from_id", "to_id", "option", "segment", "transport_mode", "departure_time",
             "distance", "travel_time", "wait_time", "feed", "agency_id", "route_id",
             "start_stop_id", "end_stop_id", "geometry"]

        Returns:
            List[Dict[str, Any]]: ordered list of segments for the chosen itinerary option
        """
        segments: List[Dict[str, Any]] = []

        try:
            # Normalize to pandas/GeoDataFrame if possible
            # (r5py DetailedItineraries often returns a GeoDataFrame-like object)
            if journey_df is None:
                return segments

            # If polars or custom object: try to convert to pandas
            to_pandas = getattr(journey_df, "to_pandas", None)
            if callable(to_pandas):
                try:
                    journey_pd = to_pandas()
                except Exception:
                    journey_pd = journey_df  # fallback to original
            else:
                journey_pd = journey_df

            # If it's a GeoDataFrame-like, ensure it's a pandas DataFrame for row ops
            if isinstance(journey_pd, gpd.GeoDataFrame) or hasattr(
                journey_pd, "columns"
            ):
                # use as-is
                pass
            else:
                # Try to coerce to pandas.DataFrame
                try:
                    journey_pd = pd.DataFrame(journey_pd)
                except Exception:
                    # give up and return empty
                    return segments

            # Empty check
            if getattr(journey_pd, "empty", False) or len(journey_pd) == 0:
                return segments

            # Choose the "best" itinerary option:
            # r5py often returns an 'option' column; pick the option value from the first row.
            first_row = None
            try:
                first_row = journey_pd.iloc[0]
            except Exception:
                # if .iloc fails, try alternative access
                try:
                    first_row = journey_pd.row(0, named=True)
                except Exception:
                    # last resort: convert to pandas again
                    journey_pd = pd.DataFrame(journey_pd)
                    if len(journey_pd) == 0:
                        return segments
                    first_row = journey_pd.iloc[0]

            chosen_option = None
            if "option" in journey_pd.columns:
                try:
                    chosen_option = int(first_row.get("option", first_row["option"]))
                except Exception:
                    # fallback: take the smallest option number if present
                    try:
                        chosen_option = int(journey_pd["option"].min())
                    except Exception:
                        chosen_option = None

            # CASE A: nested segments field in the first row (named 'segment' or 'segments')
            segments_field = None
            if "segments" in journey_pd.columns:
                segments_field = (
                    first_row.get("segments", None)
                    if hasattr(first_row, "get")
                    else first_row["segments"]
                )
            elif "segment" in journey_pd.columns:
                segments_field = (
                    first_row.get("segment", None)
                    if hasattr(first_row, "get")
                    else first_row["segment"]
                )

            if segments_field:
                # segments_field should be an iterable of segment dicts/objects
                for seg in segments_field:
                    # allow seg to be dict-like or object with attributes
                    def seg_get(k, default=None):
                        if isinstance(seg, dict):
                            return seg.get(k, default)
                        if hasattr(seg, "get"):
                            return seg.get(k, default)
                        return getattr(seg, k, default)

                    seg_mode = seg_get("mode", seg_get("transport_mode", "WALK"))
                    duration = seg_get("duration", seg_get("travel_time", 0))
                    distance = seg_get("distance", 0)
                    geom_val = seg_get("geometry", None)
                    dep = seg_get("departure_time", None)
                    arr = seg_get("arrival_time", None)

                    # parse WKT geometry if string
                    geom = None
                    if geom_val is not None:
                        try:
                            if isinstance(geom_val, str):
                                geom = wkt.loads(geom_val)
                            else:
                                geom = geom_val
                        except Exception:
                            geom = None

                    # Convert duration to seconds if it's a timedelta
                    duration_seconds = 0.0
                    if duration is not None:
                        if hasattr(duration, "total_seconds"):
                            duration_seconds = float(duration.total_seconds())
                        else:
                            duration_seconds = float(duration)

                    segments.append(
                        {
                            "mode": (
                                str(seg_mode).upper()
                                if seg_mode is not None
                                else "WALK"
                            ),
                            "duration": duration_seconds,  # Always in seconds
                            "distance": (
                                float(distance) if distance is not None else 0.0
                            ),
                            "geometry": geom,
                            "departure_time": dep,
                            "arrival_time": arr,
                        }
                    )

                # Process segments for validation and fixes
                segments = self._validate_and_fix_segments(segments)
                return segments

            # CASE B: each row is a segment. Filter rows by chosen_option if available.
            # If chosen_option is None, use all rows but prefer rows near the first row's 'option'.
            df_segments = journey_pd
            if chosen_option is not None and "option" in df_segments.columns:
                try:
                    df_segments = df_segments[df_segments["option"] == chosen_option]
                except Exception:
                    # fallback to original df
                    df_segments = journey_pd

            # If there are no rows after filtering, return empty
            if getattr(df_segments, "empty", False) or len(df_segments) == 0:
                return segments

            # Build segments list from rows. Keep order by departure_time if present, else by index.
            if "departure_time" in df_segments.columns:
                try:
                    df_segments = df_segments.sort_values("departure_time")
                except Exception:
                    pass

            # Iterate rows and map columns to our segment format
            for _, row in df_segments.iterrows():
                try:
                    mode = None
                    if "transport_mode" in df_segments.columns:
                        mode = (
                            row.get("transport_mode", None)
                            if hasattr(row, "get")
                            else row["transport_mode"]
                        )
                    if not mode and "transport_mode" in row:
                        mode = row["transport_mode"]
                    if not mode:
                        mode = (
                            row.get("mode", "WALK")
                            if hasattr(row, "get")
                            else row.get("mode", "WALK")
                        )

                    duration = None
                    if "travel_time" in df_segments.columns:
                        duration = (
                            row.get("travel_time", None)
                            if hasattr(row, "get")
                            else row["travel_time"]
                        )
                    if duration is None and "duration" in df_segments.columns:
                        duration = (
                            row.get("duration", 0)
                            if hasattr(row, "get")
                            else row["duration"]
                        )

                    distance = (
                        row.get("distance", None)
                        if "distance" in df_segments.columns
                        else None
                    )
                    geom_val = (
                        row.geometry
                        if hasattr(row, "geometry")
                        else (
                            row.get("geometry", None) if hasattr(row, "get") else None
                        )
                    )
                    dep = (
                        row.get("departure_time", None)
                        if "departure_time" in df_segments.columns
                        else None
                    )
                    arr = (
                        row.get("arrival_time", None)
                        if "arrival_time" in df_segments.columns
                        else None
                    )

                    # parse geometry if WKT string
                    geom = None
                    if geom_val is not None:
                        try:
                            if isinstance(geom_val, str):
                                geom = wkt.loads(geom_val)
                            else:
                                geom = geom_val
                        except Exception:
                            geom = None

                    # Convert duration to seconds
                    duration_seconds = 0.0
                    if duration is not None:
                        if hasattr(duration, "total_seconds"):
                            duration_seconds = float(duration.total_seconds())
                        else:
                            duration_seconds = float(duration)

                    segments.append(
                        {
                            "mode": str(mode).upper() if mode is not None else "WALK",
                            "duration": duration_seconds,  # Always in seconds
                            "distance": (
                                float(distance) if distance is not None else 0.0
                            ),
                            "geometry": geom,
                            "departure_time": dep,
                            "arrival_time": arr,
                        }
                    )
                except Exception:
                    # skip a problematic row but continue processing others
                    traceback.print_exc()
                    continue

            # Process segments for validation and fixes
            segments = self._validate_and_fix_segments(segments)

            # After processing segments, identify waiting periods
            for i, segment in enumerate(segments):
                if segment.get("mode") == "TRANSIT" and i > 0:
                    # Check if there's a waiting period before this transit segment
                    prev_segment = segments[i - 1]
                    if prev_segment.get("mode") == "WALK" and "end_stop_id" in segment:
                        # This walk segment likely represents waiting at a stop
                        prev_segment["mode"] = "WAIT"
                        prev_segment["waiting_for"] = segment.get("route_id", "TRANSIT")

            return segments

        except Exception as e:
            print(f"Error processing R5py journey: {e}")
            traceback.print_exc()
            return []

    def _validate_and_fix_segments(
        self, segments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Validate and fix journey segments for realistic simulation.

        Args:
            segments: List of journey segments

        Returns:
            List of validated and fixed segments
        """
        fixed_segments = []

        for i, segment in enumerate(segments):
            # CRITICAL FIX: Ensure durations are realistic and in seconds
            duration = segment.get("duration", 0)
            mode = segment.get("mode", "WALK").upper()
            distance = segment.get("distance", 0)

            if duration <= 0:
                # Set minimum realistic durations based on mode and distance
                if mode in ["WALK", "WALKING"]:
                    # Walking: ~5 km/h = 1.4 m/s
                    if distance > 0:
                        duration = max(distance / 1.4, 30)  # Min 30 seconds
                    else:
                        duration = 120  # Default 2 minutes
                elif mode in ["TRANSIT", "BUS", "TRAIN", "METRO", "SUBWAY"]:
                    # Transit: estimate based on distance or set minimum
                    if distance > 0:
                        duration = max(
                            distance / 10, 300
                        )  # Min 5 minutes for transit (~36 km/h average)
                    else:
                        duration = 600  # Default 10 minutes
                elif mode in ["WAIT", "TRANSIT_WAIT"]:
                    duration = 300  # Default 5 minutes waiting
                elif mode == "BIKE":
                    # Cycling: ~15 km/h = 4.2 m/s
                    if distance > 0:
                        duration = max(distance / 4.2, 60)  # Min 1 minute
                    else:
                        duration = 300  # Default 5 minutes
                else:
                    duration = 120  # Default 2 minutes for unknown modes

                segment["duration"] = duration
                print(
                    f"Fixed duration for segment {i+1} ({mode}): {duration:.0f} seconds"
                )

            # Ensure minimum duration for very short segments
            if duration < 10:
                segment["duration"] = 10  # Minimum 10 seconds
                print(
                    f"Increased minimum duration for segment {i+1} ({mode}): 10 seconds"
                )

            # FIXED: Ensure geometry endpoints align with journey start/end
            if i == len(segments) - 1:  # Final segment
                # Verify final segment leads to destination
                geometry = segment.get("geometry")
                if geometry and hasattr(geometry, "coords"):
                    coords = list(geometry.coords)
                    if coords:
                        final_point = coords[-1]  # (lon, lat)
                        segment["final_destination"] = (
                            final_point[1],
                            final_point[0],
                        )  # (lat, lon)
                        print(
                            f"Final segment destination: {segment['final_destination']}"
                        )

            # Add segment validation info
            segment["segment_index"] = i
            segment["is_final_segment"] = i == len(segments) - 1

            fixed_segments.append(segment)

        # Calculate total journey time
        total_duration = sum(seg.get("duration", 0) for seg in fixed_segments)
        print(
            f"Total journey duration: {total_duration:.0f} seconds ({total_duration/60:.1f} minutes)"
        )

        return fixed_segments

    # --- HELPER METHODS (API FOR AGENTS) ---

    def is_pos_in_evacuation_area(
        self, pos: Tuple[float, float], if_lat_lon: bool = True
    ) -> bool:
        """
        Check if position is inside evacuation polygon.

        Args:
            pos (Tuple[float, float]): Position coordinates
            if_lat_lon (bool): True if pos is (lat, lon), False if (lon, lat)

        Returns:
            bool: True if position is inside evacuation area
        """
        if if_lat_lon:
            # Convert (lat, lon) to (lon, lat) for Shapely
            lon, lat = pos[1], pos[0]
        else:
            # Already in (lon, lat) format
            lon, lat = pos[0], pos[1]

        return self.evac_polygon.contains(Point(lon, lat))

    def is_node_in_evacuation_area(self, node_id: int, mode: str) -> bool:
        """
        Check if graph node is inside evacuation area.

        Args:
            node_id (int): Graph node identifier
            mode (str): Transportation mode

        Returns:
            bool: True if node is inside evacuation area
        """
        mode = self._normalize_mode(mode)
        graph = self.graphs.get(mode)
        if not graph or node_id not in graph.nodes:
            return False

        node_data = graph.nodes[node_id]
        node_lat, node_lon = node_data.get("y"), node_data.get("x")

        if node_lat is None or node_lon is None:
            return False

        return self.is_pos_in_evacuation_area((node_lat, node_lon), if_lat_lon=True)

    def get_nearest_node(self, pos: Tuple[float, float], mode: str) -> Optional[int]:
        """
        Find nearest graph node to given position.

        Args:
            pos (Tuple[float, float]): Position as (lat, lon)
            mode (str): Transportation mode

        Returns:
            Optional[int]: Nearest node ID or None
        """
        mode = self._normalize_mode(mode)
        graph = self.graphs.get(mode)

        # Fall back to walking graph if requested mode not available
        if graph is None:
            graph = self.graphs.get("WALKING")
            if graph is None:
                return None

        if pos is None:
            return None

        # Build KDTree for fast nearest node search
        node_points = []
        node_ids = []
        for node_id, data in graph.nodes(data=True):
            if "x" in data and "y" in data:
                try:
                    x, y = float(data["x"]), float(data["y"])
                    # OSMnx: x=longitude, y=latitude -> convert to (lat, lon)
                    node_points.append((y, x))
                    node_ids.append(node_id)
                except (ValueError, TypeError):
                    continue

        if not node_points:
            return None

        # Find nearest node
        from scipy.spatial import KDTree

        kdtree = KDTree(node_points)
        dist, idx = kdtree.query([pos], k=1)
        return node_ids[idx[0]]

    def get_nearest_shelter_node(
        self, agent: EvacuationAgent, source_node: int, mode: str
    ) -> Optional[int]:
        """
        Find nearest shelter using Dijkstra single-source shortest path.

        Args:
            agent (EvacuationAgent): Requesting agent (for logging)
            source_node (int): Starting node ID
            mode (str): Transportation mode

        Returns:
            Optional[int]: Nearest shelter node ID or None
        """
        shelter_start = time.time()
        mode = self._normalize_mode(mode)
        shelters = self.shelter_nodes.get(mode)

        if not shelters or source_node is None:
            shelter_time = time.time() - shelter_start
            print(
                f"No shelters available for agent {agent.unique_id} ({mode}) - {shelter_time:.3f}s"
            )
            return None

        graph = self.graphs.get(mode)
        if not graph:
            # Fall back to walking graph
            mode = "WALKING"
            graph = self.graphs.get(mode)
            shelters = self.shelter_nodes.get(mode)
            if not graph:
                return None

        # Check if already at a shelter
        if source_node in shelters:
            shelter_time = time.time() - shelter_start
            print(f"Agent {agent.unique_id} already at shelter - {shelter_time:.3f}s")
            return source_node

        try:
            # Single Dijkstra run to find distances to all reachable nodes
            distances = nx.single_source_dijkstra_path_length(
                graph, source_node, weight="length"
            )

            # Find nearest reachable shelter
            min_distance = float("inf")
            nearest_shelter = None

            for shelter in shelters:
                if shelter in distances and distances[shelter] < min_distance:
                    min_distance = distances[shelter]
                    nearest_shelter = shelter

            # Validate shelter is outside evacuation area
            if nearest_shelter and self.is_node_in_evacuation_area(
                nearest_shelter, mode
            ):
                print(
                    f"Warning: Shelter node {nearest_shelter} is inside evacuation area"
                )
                return None

            shelter_time = time.time() - shelter_start
            if nearest_shelter:
                print(
                    f"Shelter found for agent {agent.unique_id}: {min_distance:.0f}m - {shelter_time:.3f}s"
                )
            else:
                print(
                    f"No reachable shelter for agent {agent.unique_id} - {shelter_time:.3f}s"
                )

            return nearest_shelter

        except (nx.NetworkXError, Exception) as e:
            shelter_time = time.time() - shelter_start
            print(
                f"Shelter search failed for agent {agent.unique_id}: {e} - {shelter_time:.3f}s"
            )
            return None

    def get_node_coordinates(
        self, node_id: int, mode: str
    ) -> Optional[Tuple[float, float]]:
        """
        Get coordinates of a graph node.

        Args:
            node_id (int): Node identifier
            mode (str): Transportation mode

        Returns:
            Optional[Tuple[float, float]]: (lat, lon) coordinates or None
        """
        mode = self._normalize_mode(mode)
        graph = self.graphs.get(mode)

        if not graph or node_id not in graph.nodes:
            return None

        node_data = graph.nodes[node_id]
        if "x" in node_data and "y" in node_data:
            # OSMnx: x=lon, y=lat -> return as (lat, lon)
            return (node_data["y"], node_data["x"])

        return None

    def get_edge_data(
        self, u: int, v: int, graph: nx.Graph
    ) -> Optional[Dict[str, Any]]:
        """
        Get edge data from graph, handling both simple and multigraphs.

        Args:
            u (int): Source node
            v (int): Target node
            graph (nx.Graph): Network graph

        Returns:
            Optional[Dict[str, Any]]: Edge attributes or None
        """
        try:
            if graph.is_multigraph():
                if graph.has_edge(u, v):
                    # Return first edge data for multigraphs
                    edge_keys = list(graph[u][v].keys())
                    if edge_keys:
                        return graph[u][v][edge_keys[0]]
            else:
                if graph.has_edge(u, v):
                    return graph[u][v]
        except (KeyError, AttributeError):
            pass
        return None

    def plan_route_astar(
        self, agent: EvacuationAgent, source_node: int, target_node: int
    ) -> List[int]:
        """
        Calculate shortest path using A* algorithm with geographic heuristic.

        Args:
            agent (EvacuationAgent): Requesting agent
            source_node (int): Starting node
            target_node (int): Destination node

        Returns:
            List[int]: Path as list of node IDs
        """
        mode = self._normalize_mode(agent.main_mode)
        graph = self.graphs.get(mode)

        # Fall back to walking graph
        if graph is None:
            mode = "WALKING"
            graph = self.graphs.get(mode)
            if graph is None:
                return []

        if source_node is None or target_node is None:
            return []

        if source_node == target_node:
            return [source_node]

        # Get target coordinates for heuristic
        target_data = graph.nodes.get(target_node)
        if not target_data or "x" not in target_data or "y" not in target_data:
            return self._plan_route_dijkstra(agent, source_node, target_node, graph)

        target_x, target_y = float(target_data["x"]), float(target_data["y"])

        def heuristic(u: int, v: int) -> float:
            """Geographic heuristic using haversine distance."""
            node_data = graph.nodes.get(u)
            if not node_data or "x" not in node_data or "y" not in node_data:
                return 0.0

            node_x, node_y = float(node_data["x"]), float(node_data["y"])

            # Calculate haversine distance (OSMnx: x=lon, y=lat)
            node_coords = (node_y, node_x)  # (lat, lon)
            target_coords = (target_y, target_x)  # (lat, lon)

            distance = haversine(node_coords, target_coords, unit="m")

            # Convert to travel time estimate for better heuristic
            if mode == "CAR":
                return distance / max(agent.speed_m_s, 0.1) * 1.1
            else:
                return distance

        def weight_function(u, v, data):
            """Dynamic weight considering congestion for cars."""
            base_length = data.get("length", 1.0)

            if mode == "CAR":
                # Apply congestion penalty
                base_time = base_length / max(agent.speed_m_s, 0.1)
                congestion = self.get_edge_congestion((u, v))
                penalty = 2**congestion
                return base_time * penalty
            else:
                return base_length

        try:
            path = nx.astar_path(
                graph,
                source=source_node,
                target=target_node,
                heuristic=heuristic,
                weight=weight_function,
            )
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound, KeyError):
            return []

    def _plan_route_dijkstra(
        self,
        agent: EvacuationAgent,
        source_node: int,
        target_node: int,
        graph: nx.Graph,
    ) -> List[int]:
        """Fallback routing using Dijkstra algorithm."""
        try:

            def weight_function(u, v, data):
                base_length = data.get("length", 1.0)
                if agent.main_mode == "CAR":
                    base_time = base_length / max(agent.speed_m_s, 0.1)
                    congestion = self.get_edge_congestion((u, v))
                    return base_time * (2**congestion)
                return base_length

            return nx.shortest_path(
                graph, source=source_node, target=target_node, weight=weight_function
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def report_edge_usage(self, agent: EvacuationAgent, edge: Tuple[int, int]) -> None:
        """
        Report edge usage for traffic congestion monitoring.

        Args:
            agent (EvacuationAgent): Agent using the edge
            edge (Tuple[int, int]): Edge as (u, v) node pair
        """
        if agent.main_mode == "CAR":
            self.edge_load[edge] += 1
            self.edge_agents[edge].append(agent)

    def get_edge_congestion(self, edge: Tuple[int, int]) -> float:
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
        return mode_mapping.get(mode, "WALKING")  # Default to walking

    def collect_agent_paths_data(self) -> pl.DataFrame:
        """
        Collect path history data from all agents into the Polars DataFrame.
        FIXED: Proper status validation, position tracking, and evacuation area verification.
        """
        print("Collecting agent path history data...")

        # Create lists for each column
        agent_ids = []
        svi_values = []
        original_modes = []
        final_modes = []
        statuses = []
        evac_times = []
        start_lats = []
        start_lons = []
        end_lats = []
        end_lons = []
        fail_reasons = []
        started_ats = []
        arrived_ats = []
        used_public_transports = []
        segments_completed = []

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
            # Initialize variables for this agent
            start_lat, start_lon, end_lat, end_lon = None, None, None, None
            started_at, arrived_at = None, None
            used_public_transport = getattr(agent, "using_public_transport", False)
            original_mode = getattr(
                agent, "original_mode", getattr(agent, "main_mode", "UNKNOWN")
            )
            final_mode = getattr(agent, "main_mode", "UNKNOWN")
            segments_completed_count = 0

            # CRITICAL FIX: Validate agent status before data collection
            self._validate_agent_final_status(agent)

            # Process path history if available
            if hasattr(agent, "path_history") and agent.path_history:
                print(
                    f"Agent {agent.id}: Has {len(agent.path_history)} path history entries"
                )

                # Write individual trace CSV for this agent
                if len(agent.path_history) >= 1:
                    try:
                        # Add additional debugging info to trace
                        enriched_history = []
                        for entry in agent.path_history:
                            enriched_entry = entry.copy()
                            # Add evacuation area status for each position
                            if entry.get("y") and entry.get("x"):
                                pos = (entry["y"], entry["x"])  # (lat, lon)
                                enriched_entry["in_evacuation_area"] = (
                                    self.is_pos_in_evacuation_area(pos, if_lat_lon=True)
                                )
                            else:
                                enriched_entry["in_evacuation_area"] = None
                            enriched_history.append(enriched_entry)

                        his_df = pl.DataFrame(enriched_history)
                        trace_file = f"{traces_dir}/{agent.unique_id}.csv"
                        his_df.write_csv(trace_file)
                        print(
                            f"Written trace for agent {agent.unique_id} ({len(agent.path_history)} entries)"
                        )
                    except Exception as e:
                        print(f"Error writing trace for agent {agent.unique_id}: {e}")
                        import traceback

                        traceback.print_exc()

                # Extract start position (first entry)
                if len(agent.path_history) > 0:
                    first_entry = agent.path_history[0]
                    start_lat = first_entry.get("y")
                    start_lon = first_entry.get("x")
                    started_at = first_entry.get("time")
                    if isinstance(started_at, datetime):
                        started_at = started_at.isoformat()

                # Extract end position (last entry)
                if len(agent.path_history) > 0:
                    last_entry = agent.path_history[-1]
                    end_lat = last_entry.get("y")
                    end_lon = last_entry.get("x")
                    arrived_at = last_entry.get("time")
                    if isinstance(arrived_at, datetime):
                        arrived_at = arrived_at.isoformat()

                    # CRITICAL FIX: Verify the end position makes sense
                    if end_lat and end_lon:
                        end_pos = (end_lat, end_lon)
                        is_end_in_evac_area = self.is_pos_in_evacuation_area(
                            end_pos, if_lat_lon=True
                        )

                        # Log position validation
                        print(
                            f"Agent {agent.unique_id}: Final position ({end_lat:.6f}, {end_lon:.6f}), "
                            f"In evacuation area: {is_end_in_evac_area}, Status: {agent.status}"
                        )

                        # Check for suspicious "ARRIVED" agents still in evacuation area
                        if agent.status == "ARRIVED" and is_end_in_evac_area:
                            print(
                                f"WARNING: Agent {agent.unique_id} marked as ARRIVED but final position is in evacuation area!"
                            )
                            # Re-validate this agent's status
                            agent.status = "FAILED"
                            agent.fail_reason = "Final position verification failed - still in evacuation area"

            else:
                print(f"Agent {agent.id}: No path history available")
                # For agents with no path history, try to get their current position
                if hasattr(agent, "current_pos_node") and agent.current_pos_node:
                    coords = self.get_node_coordinates(
                        agent.current_pos_node, agent.main_mode
                    )
                    if coords:
                        start_lat, start_lon = coords
                        end_lat, end_lon = coords  # Same position if no movement

            # Calculate segments completed for multi-modal agents
            if used_public_transport and hasattr(agent, "current_journey_segment"):
                segments_completed_count = getattr(agent, "current_journey_segment", 0)
                total_segments = len(getattr(agent, "journey_plan", []))
                if total_segments > 0:
                    segments_completed_count = min(
                        segments_completed_count, total_segments
                    )

            # Append data to lists
            agent_ids.append(agent.unique_id)
            svi_values.append(getattr(agent, "svi", None))
            original_modes.append(original_mode)
            final_modes.append(final_mode)
            statuses.append(getattr(agent, "status", None))
            evac_times.append(getattr(agent, "evacuation_time", 0))
            start_lats.append(start_lat)
            start_lons.append(start_lon)
            end_lats.append(end_lat)
            end_lons.append(end_lon)
            fail_reasons.append(getattr(agent, "fail_reason", None))
            started_ats.append(started_at)
            arrived_ats.append(arrived_at)
            used_public_transports.append(used_public_transport)
            segments_completed.append(segments_completed_count)

        # Create DataFrame from collected data
        self.agent_paths_df = pl.DataFrame(
            data={
                "agent_id": agent_ids,
                "svi": svi_values,
                "original_mode": original_modes,
                "final_mode": final_modes,
                "status": statuses,
                "evacuation_time": evac_times,
                "start_lat": start_lats,
                "start_lon": start_lons,
                "end_lat": end_lats,
                "end_lon": end_lons,
                "fail_reason": fail_reasons,
                "started_at": started_ats,
                "arrived_at": arrived_ats,
                "used_public_transport": used_public_transports,
                "segments_completed": segments_completed,
            }
        )

        # Generate summary statistics
        self._generate_collection_summary()

        print(f"Collected path data for {len(agent_ids)} agents")
        return self.agent_paths_df

    def _validate_agent_final_status(self, agent) -> None:
        """
        Validate and correct agent final status based on actual position.

        Args:
            agent: Agent to validate
        """
        if agent.status == "ARRIVED":
            # Get agent's current coordinates
            final_coords = None

            # Try to get from path history first
            if hasattr(agent, "path_history") and agent.path_history:
                last_entry = agent.path_history[-1]
                if last_entry.get("y") and last_entry.get("x"):
                    final_coords = (last_entry["y"], last_entry["x"])  # (lat, lon)

            # Fallback to current node position
            if (
                not final_coords
                and hasattr(agent, "current_pos_node")
                and agent.current_pos_node
            ):
                final_coords = self.get_node_coordinates(
                    agent.current_pos_node, agent.main_mode
                )

            # Validate final position
            if final_coords:
                lat, lon = final_coords
                if self.is_pos_in_evacuation_area((lat, lon), if_lat_lon=True):
                    print(
                        f"CRITICAL: Agent {agent.unique_id} marked as ARRIVED but final position "
                        f"({lat:.6f}, {lon:.6f}) is still in evacuation area!"
                    )
                    agent.status = "FAILED"
                    agent.fail_reason = (
                        "Position validation failed - still in evacuation area"
                    )
                else:
                    print(
                        f"Validated: Agent {agent.unique_id} successfully evacuated to ({lat:.6f}, {lon:.6f})"
                    )
            else:
                print(
                    f"WARNING: Cannot validate final position for agent {agent.unique_id} - no coordinates available"
                )

        # Additional validation for multi-modal agents
        if getattr(agent, "using_public_transport", False):
            if agent.status == "ARRIVED":
                print(
                    f"Multi-modal agent {agent.unique_id}: Journey completed successfully"
                )
            elif agent.status == "EVACUATING":
                current_segment = getattr(agent, "current_journey_segment", 0)
                total_segments = len(getattr(agent, "journey_plan", []))
                print(
                    f"Multi-modal agent {agent.unique_id}: Still evacuating - segment {current_segment}/{total_segments}"
                )
            elif agent.status == "FAILED":
                print(
                    f"Multi-modal agent {agent.unique_id}: Failed - {getattr(agent, 'fail_reason', 'Unknown reason')}"
                )

    def _generate_collection_summary(self) -> None:
        """Generate and print summary statistics of collected agent data."""
        if self.agent_paths_df.is_empty():
            print("No agent data collected")
            return

        # Status distribution
        status_counts = (
            self.agent_paths_df.group_by("status")
            .agg(pl.count())
            .sort("count", descending=True)
        )
        print("\nAgent Status Distribution:")
        for row in status_counts.iter_rows(named=True):
            print(f"  {row['status']}: {row['count']}")

        # Transport mode distribution
        mode_counts = (
            self.agent_paths_df.group_by("final_mode")
            .agg(pl.count())
            .sort("count", descending=True)
        )
        print("\nTransport Mode Distribution:")
        for row in mode_counts.iter_rows(named=True):
            print(f"  {row['final_mode']}: {row['count']}")

        # Public transport usage
        pt_usage = self.agent_paths_df.group_by("used_public_transport").agg(pl.count())
        print("\nPublic Transport Usage:")
        for row in pt_usage.iter_rows(named=True):
            status = "Used PT" if row["used_public_transport"] else "No PT"
            print(f"  {status}: {row['count']}")

        # Evacuation time statistics for successful evacuees
        arrived_agents = self.agent_paths_df.filter(pl.col("status") == "ARRIVED")
        if not arrived_agents.is_empty():
            evac_stats = arrived_agents.select(pl.col("evacuation_time")).describe()
            print(f"\nEvacuation Time Statistics (successful evacuees only):")
            print(f"  Mean: {evac_stats['evacuation_time'][1]:.1f} seconds")
            print(f"  Median: {evac_stats['evacuation_time'][5]:.1f} seconds")
            print(f"  Max: {evac_stats['evacuation_time'][7]:.1f} seconds")

        # Position validation summary
        agents_with_coords = self.agent_paths_df.filter(
            (pl.col("end_lat").is_not_null()) & (pl.col("end_lon").is_not_null())
        )
        print(f"\nPosition Tracking:")
        print(
            f"  Agents with final coordinates: {len(agents_with_coords)}/{len(self.agent_paths_df)}"
        )

        # Check for agents marked as arrived but potentially in wrong location
        arrived_with_coords = agents_with_coords.filter(pl.col("status") == "ARRIVED")
        if not arrived_with_coords.is_empty():
            print(
                f"  Successfully arrived agents with coordinates: {len(arrived_with_coords)}"
            )

            # Sample a few final positions for manual verification
            sample_size = min(5, len(arrived_with_coords))
            sample = arrived_with_coords.sample(n=sample_size)
            print(f"  Sample final positions:")
            for row in sample.iter_rows(named=True):
                print(
                    f"    Agent {row['agent_id']}: ({row['end_lat']:.6f}, {row['end_lon']:.6f})"
                )


def run_simulation(parameters: Dict[str, Any]) -> Tuple[EvacuationModel, Any]:
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
