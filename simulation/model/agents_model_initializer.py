import datetime
import gc
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import polars as pl
import shapely
from haversine.haversine import haversine, Unit
from shapely.geometry import Point


class AgentsGatherer:
    def __init__(
        self,
        evac_area_center: tuple[float, float],
        evacuation_area_polygon: shapely.geometry.Polygon,
        time: str,
    ) -> None:
        self.center = evac_area_center
        self.evacuation_area_polygon = evacuation_area_polygon
        self.date_time = datetime.datetime.strptime(time, "%m:%d:%H:%M")
        self.__data_path = Path(__file__).parent.parent.parent / "data"

    @property
    def __reading_trips_df_and_gathering_their_data(self) -> pl.DataFrame:
        target_dt: datetime.datetime = self.date_time
        target_month = target_dt.month
        target_day = target_dt.day

        # read full CSV
        gps_df = pl.read_csv(
            f"{self.__data_path}/trips_dataset.csv",
            try_parse_dates=False,
            infer_schema_length=10000,
        )

        gps_df = gps_df.with_columns(
            pl.col("Date_EMG")
            .str.strptime(pl.Date, "%Y-%m-%d", strict=True, exact=True)
            .alias("Date_EMG_parsed")
        )

        gps_df = gps_df.filter(
            (pl.col("Date_EMG_parsed").dt.month() == target_month)
            & (pl.col("Date_EMG_parsed").dt.day() == target_day)
        )

        gps_df = gps_df.with_columns(
            pl.concat_str(
                [
                    pl.col("Date_O").cast(pl.Utf8),
                    pl.lit(" "),
                    pl.col("Time_O").cast(pl.Utf8),
                ]
            )
            .str.strptime(
                pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=True, exact=True
            )
            .alias("start_datetime"),
            pl.concat_str(
                [
                    pl.col("Date_D").cast(pl.Utf8),
                    pl.lit(" "),
                    pl.col("Time_D").cast(pl.Utf8),
                ]
            )
            .str.strptime(
                pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=True, exact=True
            )
            .alias("end_datetime"),
        )

        gps_df = gps_df.select(
            [
                "ID",
                "Main_Mode",
                "Mode_1",
                "Mode_2",
                "Mode_3",
                "Mode_4",
                "Mode_5",
                "Purpose_O",
                "Purpose_D",
                "start_datetime",
                "end_datetime",
            ]
        )
        gc.collect()
        return gps_df

    def read_and_summarize_agents(
        self,
        output_csv_path: str = "mesa_initializers.csv",
        fallback_to_full_trace: bool = True,
        verbose: bool = True,
    ):
        output_csv_path = self.__data_path / output_csv_path

        # Extract target components (ignoring year)
        target_month = self.date_time.month
        target_day = self.date_time.day
        target_hour = self.date_time.hour

        trips_df = self.__reading_trips_df_and_gathering_their_data
        if verbose:
            print("Total trips chosen:", trips_df.shape[0])

        chosen_trips = trips_df.select(
            ["ID", "Main_Mode", "start_datetime", "end_datetime"]
        ).to_dicts()
        del trips_df
        gc.collect()

        summaries: list[dict[str, Any]] = []
        for trip in chosen_trips:
            agent_id = trip.get("ID")
            main_mode = trip.get("Main_Mode")
            if not agent_id:
                continue

            gps_path = f"{self.__data_path}/gps_dataset/{agent_id}.csv"
            try:
                df_c = pl.read_csv(gps_path, try_parse_dates=False)
            except FileNotFoundError:
                if verbose:
                    print(f"GPS file missing for ID {agent_id}")
                continue
            agent_home = self.__get_centroid_of_his_locations(df_c)

            # Parse LOCAL DATETIME with error handling
            try:
                df_c = df_c.with_columns(
                    pl.col("LOCAL DATETIME")
                    .str.strptime(
                        pl.Datetime,
                        "%Y-%m-%d %H:%M:%S",
                        strict=True,
                        exact=True,  # THIS IS THE CORRECT FORMAT DON'T CHANGE!
                    )
                    .alias("local_dt")
                )
            except Exception as e:
                if verbose:
                    print(f"Error parsing datetime for {agent_id}: {e}")
                continue

            # Filter out rows with null datetime
            df_c = df_c.filter(pl.col("local_dt").is_not_null())

            if df_c.is_empty():
                if verbose:
                    print(f"No valid datetime records for {agent_id}")
                continue

            df_search = None
            fallback_used = False

            # Primary filter: exact month/day match with time window
            try:
                df_win = df_c.filter(
                    (pl.col("local_dt").dt.month() == target_month)
                    & (pl.col("local_dt").dt.day() == target_day)
                    & (
                        (pl.col("local_dt").dt.hour()).is_between(
                            target_hour - 4, target_hour + 4
                        )
                    )
                )

                if not df_win.is_empty():
                    df_search = df_win
                else:
                    raise ValueError("No samples in primary window")

            except Exception:
                if fallback_to_full_trace:
                    # Fallback 1: Same month/day, any time
                    try:
                        df_search = df_c.filter(
                            (pl.col("local_dt").dt.month() == target_month)
                            & (pl.col("local_dt").dt.day() == target_day)
                        )
                        if df_search.is_empty():
                            raise ValueError("No samples for same day")
                        fallback_used = True
                        if verbose:
                            print(f"Using same-day fallback for {agent_id}")
                    except Exception:
                        # Fallback 2: Same month, any day/time
                        try:
                            df_search = df_c.filter(
                                pl.col("local_dt").dt.month() == target_month
                            )
                            if df_search.is_empty():
                                raise ValueError("No samples for same month")
                            fallback_used = True
                            if verbose:
                                print(f"Using same-month fallback for {agent_id}")
                        except Exception:
                            # Fallback 3: Use entire dataset
                            df_search = df_c
                            fallback_used = True
                            if verbose:
                                print(f"Using full dataset fallback for {agent_id}")

            if df_search is None or df_search.is_empty():
                if verbose:
                    print(f"No GPS samples at all for {agent_id}")
                continue

            # Extract data lists
            try:
                lats = df_search["LATITUDE"].to_list()
                lons = df_search["LONGITUDE"].to_list()
                times = df_search["local_dt"].to_list()
                speeds_raw = (
                    df_search.get_column("SPEED").to_list()
                    if "SPEED" in df_search.columns
                    else [None] * len(lats)
                )
            except Exception as e:
                if verbose:
                    print(f"Error extracting data for {agent_id}: {e}")
                continue

            # Filter out invalid coordinates
            valid_coords = []
            valid_times = []
            valid_speeds = []

            for i, (lat, lon) in enumerate(zip(lats, lons)):
                if (
                    lat is not None
                    and lon is not None
                    and not (np.isnan(lat) or np.isnan(lon))
                ):
                    valid_coords.append((lat, lon))
                    valid_times.append(times[i])
                    valid_speeds.append(speeds_raw[i] if i < len(speeds_raw) else None)

            if not valid_coords:
                if verbose:
                    print(f"No valid coordinates for {agent_id}")
                continue

            # Clean speeds and compute statistics
            cleaned_speeds = self.__eliminate_outliers_iqr(valid_speeds)
            median_speed = (
                float(np.median(cleaned_speeds)) if len(cleaned_speeds) else None
            )
            mean_speed = float(np.mean(cleaned_speeds)) if len(cleaned_speeds) else None
            stationary_fraction = (
                (sum(1 for s in cleaned_speeds if s < 0.5) / len(cleaned_speeds))
                if len(cleaned_speeds)
                else 0.0
            )

            # Find nearest-to-center sample
            min_d = float("inf")
            best_idx = None
            for idx, (lat, lon) in enumerate(valid_coords):
                try:
                    d = self.__haversine_distance_m((lat, lon), self.center)
                    if d < min_d and self.are_coords_in_the_evacuation_area((lat, lon)):
                        min_d = d
                        best_idx = idx
                except Exception:
                    continue

            if best_idx is None:
                if verbose:
                    print(f"Could not find valid nearest point for {agent_id}")
                continue

            # Extract best sample data
            start_lat, start_lon = valid_coords[best_idx]
            start_time = valid_times[best_idx]

            summary = {
                "ID": agent_id,
                "start_time": start_time.isoformat() if start_time else None,
                "start_lat": start_lat,
                "start_lon": start_lon,
                "main_mode": main_mode,
                "median_speed_m_s": median_speed,
                "mean_speed_m_s": mean_speed,
                "n_points_window": len(valid_coords),
                "min_dist_to_center_m": min_d,
                "stationary_fraction": stationary_fraction,
                "home_location_lat": agent_home[0],
                "home_location_lon": agent_home[1],
                "used_fallback_full_trace": fallback_used,
            }
            summaries.append(summary)

        if summaries:
            base_df = pl.DataFrame(summaries)
            cleaned_df = self.__make_sure_speeds_are_correct(base_df)
            walking_speeds_df = self.__get_walking_speed_per_agent(cleaned_df)

            # Join the walking speeds back to the main dataframe
            out_df = cleaned_df.join(walking_speeds_df, on="ID", how="left")

            # Fill null walking speeds with median
            if out_df.select("walking_speed_m_s").null_count().item() != 0:
                walking_median = out_df.select("walking_speed_m_s").median().item()
                out_df = out_df.with_columns(
                    pl.col("walking_speed_m_s").fill_null(value=walking_median)
                )

            # Get cycling speeds and join them
            cycling_speeds_df = self.__get_cycling_speed_per_agent(cleaned_df)
            out_df = out_df.join(cycling_speeds_df, on="ID", how="left")

            """
            # Fill null cycling speeds with median
            if out_df.select("cycling_speed_m_s").null_count().item() != 0:
                cycling_median = out_df.select("cycling_speed_m_s").median().item()
                out_df = out_df.with_columns(
                    pl.col("cycling_speed_m_s").fill_null(value=cycling_median)
                )
            """

            svi_scores = pl.read_csv(
                f"{self.__data_path}/agents_svi_scores.csv"
            ).select("SVI_normalized", "ID")

            out_df = out_df.join(svi_scores, on="ID", how="left")

            out_df.write_csv(output_csv_path)

            del (
                out_df,
                base_df,
                cleaned_df,
                walking_speeds_df,
                cycling_speeds_df,
                svi_scores,
            )
            if verbose:
                print("Wrote initializer CSV:", output_csv_path)

        gc.collect()
        return summaries

    @staticmethod
    def __eliminate_outliers_iqr(vals: List[float]) -> List[float]:
        arr = np.array(
            [v for v in vals if v is not None and not np.isnan(v)], dtype=float
        )
        if arr.size == 0:
            return []
        q1 = np.percentile(arr, 25)
        q3 = np.percentile(arr, 75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        cleaned = arr[(arr >= lower) & (arr <= upper)]
        return cleaned.tolist()

    @staticmethod
    def __haversine_distance_m(
        point1: tuple[float, float], point2: tuple[float, float]
    ) -> float:
        return haversine(point1, point2, unit=Unit.METERS, check=True, normalize=True)

    def are_coords_in_the_evacuation_area(self, pos: tuple[float, float]):
        """Checks if a (lat, lon) point is inside the evacuation polygon."""
        return self.evacuation_area_polygon.contains(Point(pos[0], pos[1]))

    def __make_sure_speeds_are_correct(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Clean and validate speeds for each transportation mode by removing outliers.

        Args:
            df: DataFrame containing agent data with 'main_mode' and speed columns

        Returns:
            DataFrame with cleaned speed values
        """
        if df.is_empty():
            return df

        # Get unique modes
        unique_modes = df.select("main_mode").unique().to_series().to_list()

        # Process each mode separately
        cleaned_dfs = []

        for mode in unique_modes:
            if mode is None:
                continue

            # Filter data for this mode
            mode_df = df.filter(pl.col("main_mode") == mode)

            if mode_df.is_empty():
                continue

            # Extract speed values for cleaning
            if "median_speed_m_s" in mode_df.columns:
                raw_speeds = mode_df.select("median_speed_m_s").to_series().to_list()
                cleaned_speeds = self.__eliminate_outliers_iqr(raw_speeds)

                # Create mapping of original to cleaned speeds
                if cleaned_speeds:
                    # Keep only rows with speeds within the cleaned range
                    min_clean = min(cleaned_speeds)
                    max_clean = max(cleaned_speeds)

                    mode_df_cleaned = mode_df.filter(
                        pl.col("median_speed_m_s").is_between(
                            min_clean, max_clean, closed="both"
                        )
                    )
                else:
                    # If no valid speeds after cleaning, keep original data
                    mode_df_cleaned = mode_df
            else:
                # If no speed column, keep as is
                mode_df_cleaned = mode_df

            cleaned_dfs.append(mode_df_cleaned)

        gc.collect()

        if cleaned_dfs:
            result_df = pl.concat(cleaned_dfs, how="vertical")
            return result_df
        else:
            return df

    @staticmethod
    def __get_walking_speed_per_agent(df: pl.DataFrame) -> pl.DataFrame:
        """
        Get median walking speed for each agent.

        Args:
            df: DataFrame containing agent data

        Returns:
            DataFrame with ID and median walking speed per agent
        """
        walking_df = df.filter(pl.col("main_mode") == "WALKING")

        if walking_df.is_empty():
            return pl.DataFrame({"ID": [], "walking_speed_m_s": []})

        return walking_df.group_by("ID").agg(
            pl.col("median_speed_m_s").median().alias("walking_speed_m_s")
        )

    @staticmethod
    def __get_centroid_of_his_locations(df: pl.DataFrame) -> Tuple[float, float]:

        df = df.select(["LATITUDE", "LONGITUDE", "LOCAL DATETIME"])

        df = df.with_columns(
            pl.col("LOCAL DATETIME").str.strptime(
                format="%Y-%m-%d %H:%M:%S", strict=True, exact=True, dtype=pl.Datetime
            )
        )

        # Filter for nighttime hours (10 PM to 5 AM)
        df = df.filter(
            (pl.col("LOCAL DATETIME").dt.hour() >= 22)
            | (pl.col("LOCAL DATETIME").dt.hour() <= 5)
        )

        # Check if we have any data after filtering
        if df.is_empty():
            gc.collect()
            return None, None  # or raise an exception

        # Check again after coordinate filtering
        if df.is_empty():
            gc.collect()
            return None, None

        c_lat = df.select("LATITUDE").mean().item()
        c_lon = df.select("LONGITUDE").mean().item()

        del df
        gc.collect()
        return c_lat, c_lon

    @staticmethod
    def __get_cycling_speed_per_agent(df: pl.DataFrame) -> pl.DataFrame:
        """
        Get median walking speed for each agent.

        Args:
            df: DataFrame containing agent data

        Returns:
            DataFrame with ID and median walking speed per agent
        """
        walking_df = df.filter(pl.col("main_mode") == "BIKE")

        if walking_df.is_empty():
            return pl.DataFrame({"ID": [], "cycling_speed_m_s": []})

        return walking_df.group_by("ID").agg(
            pl.col("median_speed_m_s").median().alias("cycling_speed_m_s")
        )

    @staticmethod
    def __get_activity_speeds(df: pl.DataFrame) -> dict[str, float]:
        """
        Get median speeds for each transportation mode across all agents.

        Args:
            df: DataFrame containing agent data with main_mode and speed columns

        Returns:
            Dictionary mapping transportation modes to their median speeds
        """
        if df.is_empty():
            return {}

        # Group by mode and calculate median speed for each mode
        mode_speeds = (
            df.group_by("main_mode")
            .agg(pl.col("median_speed_m_s").median().alias("mode_median_speed"))
            .sort("main_mode")
        )

        # Convert to dictionary
        result = {}
        for row in mode_speeds.iter_rows(named=True):
            mode = row["main_mode"]
            speed = row["mode_median_speed"]
            if mode is not None and speed is not None:
                result[mode] = float(speed)

        return result

    @staticmethod
    def __get_speed_statistics_by_mode(df: pl.DataFrame) -> pl.DataFrame:
        """
        Get comprehensive speed statistics for each transportation mode.

        Args:
            df: DataFrame containing agent data

        Returns:
            DataFrame with mode, count, median, mean, std, min, max speeds
        """
        if df.is_empty():
            return pl.DataFrame()

        return (
            df.group_by("main_mode")
            .agg(
                [
                    pl.count("median_speed_m_s").alias("agent_count"),
                    pl.col("median_speed_m_s").median().alias("median_speed"),
                    pl.col("median_speed_m_s").mean().alias("mean_speed"),
                    pl.col("median_speed_m_s").std().alias("std_speed"),
                    pl.col("median_speed_m_s").min().alias("min_speed"),
                    pl.col("median_speed_m_s").max().alias("max_speed"),
                    pl.col("median_speed_m_s").quantile(0.25).alias("q25_speed"),
                    pl.col("median_speed_m_s").quantile(0.75).alias("q75_speed"),
                ]
            )
            .sort("main_mode")
        )

    def get_agent_speed_profiles(self, df: pl.DataFrame = None) -> dict[str, Any]:
        """
        Get comprehensive speed analysis for all agents and transportation modes.

        Args:
            df: Optional DataFrame. If None, will read from the saved CSV file.

        Returns:
            Dictionary containing various speed statistics and profiles
        """
        if df is None:
            csv_path = self.__data_path / "mesa_initializers.csv"
            try:
                df = pl.read_csv(csv_path)
            except FileNotFoundError:
                print(f"CSV file not found at {csv_path}")
                return {}

        if df.is_empty():
            return {}

        # Clean the speeds first
        cleaned_df = self.__make_sure_speeds_are_correct(df)

        return {
            "mode_statistics": self.__get_speed_statistics_by_mode(
                cleaned_df
            ).to_dicts(),
            "activity_speeds": self.__get_activity_speeds(cleaned_df),
            "walking_speeds_per_agent": self.__get_walking_speed_per_agent(
                cleaned_df
            ).to_dicts(),
            "total_agents": cleaned_df.shape[0],
            "modes_present": cleaned_df.select("main_mode")
            .unique()
            .to_series()
            .to_list(),
            "speed_range": (
                {
                    "min": float(cleaned_df.select("median_speed_m_s").min().item()),
                    "max": float(cleaned_df.select("median_speed_m_s").max().item()),
                }
                if not cleaned_df.select("median_speed_m_s").null_count().item()
                else None
            ),
        }
