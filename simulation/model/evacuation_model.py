from datetime import datetime, timedelta

import mesa
import networkx as nx
import osmnx as ox


class EvacAgent(mesa.Agent):
    def __init__(
        self,
        unique_id,
        model,
        start_lat,
        start_lon,
        departure_time: datetime,
        allowed_window_s: int,
        speed_m_s: float,
        main_mode: str,
        svi: float = 0.0,
    ):
        super().__init__(unique_id, model)
        self.start_lat = start_lat
        self.start_lon = start_lon
        self.departure_time = departure_time  # absolute datetime
        self.allowed_window_s = allowed_window_s  # e.g., 3600 for 1 hour
        self.evacuate_before = departure_time + timedelta(seconds=allowed_window_s)
        # SVI adjustments
        self.svi = float(svi)
        # e.g., vulnerable start delays or speed multipliers
        self.start_delay_seconds = int(
            self.svi * model.max_svi_start_delay_s
        )  # model param
        self.effective_departure_time = self.departure_time + timedelta(
            seconds=self.start_delay_seconds
        )
        self.deadline = self.evacuate_before  # could also add svi effect
        self.activated = False
        self.status = "idle"  # idle, evacuating, arrived, failed
        self.speed_m_s = speed_m_s * (
            1.0 - self.svi * model.svi_speed_penalty
        )  # slower if svi>0
        self.main_mode = main_mode
        # routing state
        self.current_node = model.nearest_node(self.start_lat, self.start_lon)
        self.path = []  # list of nodes to follow
        self.current_edge = None  # (u,v)
        self.edge_progress = 0.0  # meters
        self.target_node = None

    def activate_if_due(self):
        now = self.model.sim_time
        if not self.activated and now >= self.effective_departure_time:
            self.activated = True
            self.status = "evacuating"
            # decide target and plan route now or in step()
            self.assign_target_and_plan()

    def assign_target_and_plan(self):
        # Example: if inside evacuation polygon -> target nearest amenity
        if self.model.is_inside_evacuation_area(self.start_lat, self.start_lon):
            # find nearest amenity node (model has amenity_nodes list)
            self.target_node = self.model.nearest_amenity_node(
                self.start_lat, self.start_lon
            )
        else:
            # if outside evacuation, maybe go to home (stay) or go to nearest safe zone
            self.target_node = self.model.nearest_safe_node(
                self.start_lat, self.start_lon
            )
        # plan route (use agent-aware weight)
        self.path = self.model.plan_route_for_agent(
            self.current_node, self.target_node, agent=self
        )

    def step(self):
        now = self.model.sim_time
        # Activation check
        if not self.activated:
            self.activate_if_due()
            return

        # Deadline check
        if now > self.deadline and self.status != "arrived":
            # missed window
            # choose what to do: request rescue, set failed, or try alternative
            self.status = "failed"
            self.model.record_failed_agent(self)
            return

        # If arrived
        if self.status == "arrived":
            return

        # If no target/path, plan
        if (not self.path) and self.target_node:
            self.assign_target_and_plan()
            if not self.path:
                # cannot find path
                self.status = "failed"
                return

        # Move along path fractionally using step_seconds and speed
        self.move_along_path(self.model.step_seconds)

        # Check arrival
        if self.current_node == self.target_node:
            self.status = "arrived"
            self.model.record_arrival(self)

    def move_along_path(self, dt_s: float):
        # if no path or only at a node, nothing to do
        if not self.path or len(self.path) < 2:
            return
        # ensure current_edge is set
        if self.current_edge is None:
            u, v = self.path[0], self.path[1]
            self.current_edge = (u, v)
            # compute edge length
            self.edge_length = self.model.edge_length(u, v)
            self.edge_progress = 0.0

        remain = dt_s * self.speed_m_s
        while remain > 0 and self.current_edge is not None:
            left_on_edge = self.edge_length - self.edge_progress
            if remain >= left_on_edge:
                # finish edge
                remain -= left_on_edge
                # move agent to v node
                _, v = self.current_edge
                self.model.move_agent_on_graph(self, v)
                self.current_node = v
                # advance path
                if len(self.path) >= 2:
                    self.path.pop(0)
                # set next edge or finish
                if len(self.path) >= 2:
                    u, v = self.path[0], self.path[1]
                    self.current_edge = (u, v)
                    self.edge_length = self.model.edge_length(u, v)
                    self.edge_progress = 0.0
                else:
                    self.current_edge = None
            else:
                # partial progress
                self.edge_progress += remain
                remain = 0

    # helpers used by agents
    def nearest_node(self, lat, lon, graph=None):
        if graph is None:
            graph = self.G_walk
        return ox.distance.nearest_nodes(graph, X=lon, Y=lat)

    def nearest_amenity_node(self, lat, lon):
        # naive linear search or spatial index
        return min(
            self.amenity_nodes,
            key=lambda n: self.edge_length_between_node_and_latlon(n, lat, lon),
        )

    def is_inside_evacuation_area(self, lat, lon) -> bool:
        # your polygon check
        return point_in_polygon(lon, lat, self.evac_polygon_coords)

    def plan_route_for_agent(self, source_node, target_node, agent):
        # Use networkx.shortest_path with a custom weight function that uses agent.speed
        G = self.G_walk if agent.main_mode == "walking" else self.G_drive

        # define weight function using closure
        def weight(u, v, attr):
            length = attr.get("length", 0.0)
            # edge_base_speed in m/s (from attr or default)
            edge_base_speed = attr.get("speed_m_s", None)
            if edge_base_speed:
                speed = min(agent.speed_m_s, edge_base_speed)
            else:
                speed = agent.speed_m_s
            return length / speed

        try:
            path = nx.shortest_path(G, source_node, target_node, weight=weight)
            return path
        except Exception:
            return []
