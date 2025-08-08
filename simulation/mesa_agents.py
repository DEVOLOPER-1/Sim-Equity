import math
import datetime
import  numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
from haversine.haversine import haversine , Unit
EARTH_RADIUS_KM = 6371.0088
EARTH_RADIUS_M = EARTH_RADIUS_KM * 1000.0


class EnvironmentInitializer:
    """
    Initialize environment parameters for simulation.

    Parameters
    ----------
    center : tuple[lat[float], lon[float]]
        The lat, lon of the environment center point in decimal degrees.
    radius : float
        The radius of the environment boundary from the center point in KILOMETERS.
    time : str
        The simulation start time in format 'month:day:hour:minute' (e.g., '08:05:14:30')
    """

    def __init__(self, center: Tuple[float, float], radius: float, time: str) -> None:
        self.center = center
        self.radius_km = radius
        self.date_time = datetime.datetime.strptime(time, "%m:%d:%H:%M")

    # ----------------------------
    # Core geodesic helpers
    # ----------------------------
    @staticmethod
    def normalize_lon_deg(lon_deg: float) -> float:
        """Normalize longitude to [-180, 180) in degrees."""
        return ((lon_deg + 180.0) % 360.0) - 180.0

    @staticmethod
    def destination_point_spherical(lat_deg: float, lon_deg: float, bearing_rad: float, distance_km: float) -> Tuple[float, float]:
        """
        Spherical forward (direct) problem.

        Inputs:
          - lat_deg, lon_deg : start point in degrees
          - bearing_rad : initial bearing in radians (clockwise from north)
          - distance_km : distance in kilometers

        Returns:
          - (lat2_deg, lon2_deg) in degrees
        """
        lat1 = math.radians(lat_deg)
        lon1 = math.radians(lon_deg)
        delta = distance_km / EARTH_RADIUS_KM  # angular distance in radians

        lat2 = math.asin(math.sin(lat1) * math.cos(delta) +
                         math.cos(lat1) * math.sin(delta) * math.cos(bearing_rad))

        lon2 = lon1 + math.atan2(math.sin(bearing_rad) * math.sin(delta) * math.cos(lat1),
                                 math.cos(delta) - math.sin(lat1) * math.sin(lat2))

        lat2_deg = math.degrees(lat2)
        lon2_deg = math.degrees(lon2)
        lon2_deg = EnvironmentInitializer.normalize_lon_deg(lon2_deg)
        return lat2_deg, lon2_deg

    # ----------------------------
    # Public API: polygon generation
    # ----------------------------
    def calculate_evacuation_area(self, radius_km: float = None, center: Tuple[float, float] = None, points: int = 360) -> List[Tuple[float, float]]:
        """
        Build a polygon (list of lat, lon in degrees) approximating a circle on the sphere.

        - radius_km: radius in kilometers (defaults to self.radius_km)
        - center: (lat, lon) in degrees (defaults to self.center)
        - points: number of vertices around the circle
        """
        if radius_km is None:
            radius_km = self.radius_km
        if center is None:
            center = self.center

        lat0, lon0 = center
        polygon: List[Tuple[float, float]] = []
        for k in range(points):
            bearing_rad = math.radians(k * 360.0 / points) # adds tolerance if points aren't 360
            lat2_deg, lon2_deg = self.destination_point_spherical(lat0, lon0, bearing_rad, radius_km)
            polygon.append((lat2_deg, lon2_deg))
        # Close polygon (optional)
        if polygon:
            polygon.append(polygon[0])
        return polygon

    # ----------------------------
    # Utility: compute minimum points for chord length L (meters)
    # ----------------------------
    @staticmethod
    def needed_points_for_max_chord(radius_m: float, max_chord_m: float) -> int:
        """
        Choose number of points n so the maximum chord length between adjacent vertices <= max_chord_m.
        """
        if radius_m <= 0 or max_chord_m <= 0:
            return 36
        ratio = max_chord_m / (2.0 * radius_m)
        if ratio >= 1.0:
            return 4
        n = math.pi / math.asin(ratio)
        return max(4, math.ceil(n))

    # ----------------------------
    # Haversine distance (meters)
    # ----------------------------
    @staticmethod
    def haversine_distance_m(point1:tuple[float,float] , point2:tuple[float,float]) -> float:
            return haversine(point1 , point2 , unit=Unit.METERS , check=True , normalize=True)

    # ----------------------------
    # Plot helpers
    # ----------------------------
    def latlon_to_local_xy(self, lat_deg: float, lon_deg: float, center_lat_deg: float, center_lon_deg: float) -> Tuple[float, float]:
        """
        Convert lat/lon differences to local tangential plane meters (east, north),
        using the center as origin. Good for small areas (up to a few 10s km).
        """
        phi0 = math.radians(center_lat_deg)
        dlat_rad = math.radians(lat_deg - center_lat_deg)
        dlon_rad = math.radians(lon_deg - center_lon_deg)
        x = EARTH_RADIUS_M * dlon_rad * math.cos(phi0)  # East (meters)
        y = EARTH_RADIUS_M * dlat_rad  # North (meters)
        return x, y

    def plot_evacuation_area(self, points: int = 360, title: str = None, figsize: tuple = (10, 10)) -> None:
        """
        Plot evacuation area in local metric coordinates (appears as perfect circle).
        """
        if title is None:
            title = f"Evacuation Zone - {self.radius_km} km radius"

        # Generate polygon and convert to local coordinates
        polygon = self.calculate_evacuation_area(self.radius_km, self.center, points)
        center_lat, center_lon = self.center

        xy_coords = [self.latlon_to_local_xy(lat, lon, center_lat, center_lon)
                     for lat, lon in polygon]
        xs, ys = zip(*xy_coords)

        # Create plot with enhanced styling
        fig, ax = plt.subplots(figsize=figsize)

        # Plot filled area with gradient-like effect
        ax.fill(xs, ys, alpha=0.3, color='crimson', label=f'Evacuation Area ({self.radius_km} km)',
                edgecolor='darkred', linewidth=2.5)

        # Add center point with enhanced styling
        ax.plot(0, 0, marker='o', color='black', markersize=10,
                markerfacecolor='gold', markeredgewidth=2,
                label='Emergency Center', zorder=5)

        # Add radius indicators (optional grid circles)
        for r_frac in [0.25, 0.5, 0.75]:
            circle_r = self.radius_km * 1000 * r_frac  # Convert to meters
            theta = np.linspace(0, 2 * np.pi, 100)
            circle_x = circle_r * np.cos(theta)
            circle_y = circle_r * np.sin(theta)
            ax.plot(circle_x, circle_y, '--', color='gray', alpha=0.4, linewidth=1)

        # Styling and formatting
        ax.set_xlabel('Easting (meters)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Northing (meters)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_aspect('equal')
        ax.legend(loc='upper right', framealpha=0.9, fontsize=11)

        # Add coordinate info as text
        max_dist = max(max(xs), max(ys))
        ax.text(0.02, 0.98, f'Center: ({center_lat:.4f}°, {center_lon:.4f}°) \n  Max Distance{max_dist:.2f} m',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                fontsize=10)

        plt.tight_layout()
        plt.show()

    def plot_latlon_deg(self, points: int = 360, title: str = None, figsize: tuple = (12, 8)) -> None:
        """
              Plot polygon in lat/lon degrees (will look elliptical visually because degrees
              are not equal-length in E-W vs N-S). Useful for quick lat/lon inspection.
        """
        if title is None:
            title = f"Evacuation Zone - Geographic View ({self.radius_km} km radius)"

        polygon = self.calculate_evacuation_area(self.radius_km, self.center, points)
        lats, lons = zip(*polygon)

        # Calculate bounds for better display
        lat_range = max(lats) - min(lats)
        lon_range = max(lons) - min(lons)

        # Create plot with enhanced styling
        fig, ax = plt.subplots(figsize=figsize)

        # Plot filled area
        ax.fill(lons, lats, alpha=0.25, color='crimson',
                edgecolor='darkred', linewidth=2.5,
                label=f'Evacuation Boundary ({self.radius_km} km)')

        # Plot center point
        ax.plot(self.center[1], self.center[0], marker='o', color='black',
                markersize=12, markerfacecolor='gold', markeredgewidth=2,
                label='Emergency Center', zorder=5)

        # Styling and formatting
        ax.set_xlabel('Longitude (degrees)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Latitude (degrees)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_aspect('equal')
        ax.legend(loc='best', framealpha=0.9, fontsize=11)

        # Add coordinate bounds information
        info_text = (f'Bounds:\n'
                     f'Lat: {min(lats):.5f}° to {max(lats):.5f}°\n'
                     f'Lon: {min(lons):.5f}° to {max(lons):.5f}°\n'
                     f'Center: {self.center[0]:.5f}°, {self.center[1]:.5f}°')

        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))

        # Add margin for better visibility
        margin_lat = lat_range * 0.1
        margin_lon = lon_range * 0.1
        ax.set_xlim(min(lons) - margin_lon, max(lons) + margin_lon)
        ax.set_ylim(min(lats) - margin_lat, max(lats) + margin_lat)

        plt.tight_layout()
        plt.show()
