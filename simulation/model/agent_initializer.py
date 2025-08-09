import datetime
import gc
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import polars as pl
from haversine.haversine import haversine, Unit


class AgentsGatherer:
    def __init__(self, polygon_vertices: List[Tuple[float, float]], time: str) -> None:
        self.polygon_vertices = polygon_vertices
        self.center = self.polygon_vertices[-1]
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
            .str.strptime(pl.Date, "%Y-%m-%d", strict=True)
            .alias("Date_EMG_parsed")
        )
        """
        print("Unique months in data:")
        print(gps_df.select(pl.col("Date_EMG_parsed").dt.month().unique().sort()))

        print("Unique days in data:")
        print(gps_df.select(pl.col("Date_EMG_parsed").dt.day().unique().sort()))
        print(
            gps_df.select(
                pl.col("Date_EMG_parsed").value_counts().sort(descending=True)
            )
        )

        print(gps_df.shape)
        """
        gps_df = gps_df.filter(
            (pl.col("Date_EMG_parsed").dt.month() == target_month)
            & (pl.col("Date_EMG_parsed").dt.day() == target_day)
        )
        # print(gps_df.shape)

        gps_df = gps_df.with_columns(
            pl.concat_str(
                [
                    pl.col("Date_O").cast(pl.Utf8),
                    pl.lit(" "),
                    pl.col("Time_O").cast(pl.Utf8),
                ]
            )
            .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False)
            .alias("start_datetime"),
            pl.concat_str(
                [
                    pl.col("Date_D").cast(pl.Utf8),
                    pl.lit(" "),
                    pl.col("Time_D").cast(pl.Utf8),
                ]
            )
            .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False)
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
        hours_window: float = 6,
        fallback_to_full_trace: bool = True,
        verbose: bool = True,
    ):
        output_csv_path = self.__data_path / output_csv_path
        max_time = self.date_time + datetime.timedelta(hours=hours_window)
        min_time = self.date_time - datetime.timedelta(hours=hours_window)

        trips_df = self.__reading_trips_df_and_gathering_their_data
        if verbose:
            print("Total trips rows:", trips_df.shape[0])

        chosen_trips = trips_df.select(
            ["ID", "Main_Mode", "start_datetime", "end_datetime"]
        ).to_dicts()
        del trips_df
        gc.collect()

        summaries: set[dict[str, Any]] = set()
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

            # parse LOCAL DATETIME (non-strict to be tolerant)
            df_c = df_c.with_columns(
                pl.col("LOCAL DATETIME")
                .str.strptime(pl.Datetime, "%Y-%m-%d-%H-%M-%S", strict=False)
                .alias("local_dt")
            )

            # 1) windowed samples
            df_win = df_c.filter(pl.col("local_dt").is_between(min_time, max_time))

            # If no windowed samples, optionally fallback to full trace nearest point
            if df_win.is_empty() and fallback_to_full_trace:
                if verbose:
                    print(
                        f"No samples in window for {agent_id}; falling back to full trace nearest point"
                    )
                df_search = df_c  # search whole file
            else:
                df_search = df_win

            if df_search.is_empty():
                # still empty: skip
                if verbose:
                    print(f"No GPS samples at all for {agent_id}")
                continue

            # Extract lists
            lats = df_search["LATITUDE"].to_list()
            lons = df_search["LONGITUDE"].to_list()
            times = df_search["local_dt"].to_list()
            speeds_raw = (
                df_search.get_column("SPEED").to_list()
                if "SPEED" in df_search.columns
                else [None] * len(lats)
            )

            # Clean speeds (IQR) and compute stats (use your static)
            cleaned_speeds = self.__eliminate_outliers_iqr(speeds_raw)
            median_speed = (
                float(np.median(cleaned_speeds)) if len(cleaned_speeds) else None
            )
            mean_speed = float(np.mean(cleaned_speeds)) if len(cleaned_speeds) else None
            stationary_fraction = (
                (sum(1 for s in cleaned_speeds if s < 0.5) / len(cleaned_speeds))
                if len(cleaned_speeds)
                else 0.0
            )

            # Find nearest-to-center sample within df_search
            min_d = float("inf")
            best_idx = None
            for idx, (lat, lon) in enumerate(zip(lats, lons)):
                if lat is None or lon is None:
                    continue
                d = self.__haversine_distance_m((lat, lon), self.center)
                if d < min_d:
                    min_d = d
                    best_idx = idx
                    # Remove the main_mode assignment from here

            if best_idx is None:
                continue

            # choose starting point (closest sample)
            start_lat = lats[best_idx]
            start_lon = lons[best_idx]
            start_time = times[best_idx]

            # Fix the mode inference logic
            inferred_mode = None
            for current_trip in chosen_trips:  # Use different variable name
                if (
                    current_trip.get("start_datetime")
                    <= start_time
                    <= current_trip.get("end_datetime")
                ):
                    inferred_mode = current_trip.get("Main_Mode")
                    break

            # Keep the original trip's main_mode (already set at loop start)
            # main_mode is already correctly set from: main_mode = trip.get("Main_Mode")

            summary = {
                "ID": agent_id,
                "start_time": start_time.isoformat() if start_time else None,
                "start_lat": start_lat,
                "start_lon": start_lon,
                "inferred_mode": inferred_mode,
                "main_mode": main_mode,
                "median_speed_m_s": median_speed,
                "mean_speed_m_s": mean_speed,
                "n_points_window": len(lats),
                "min_dist_to_center_m": min_d,
                "stationary_fraction": stationary_fraction,
                "used_fallback_full_trace": bool(df_win.is_empty()),
            }
            summaries.add(summary)

        if summaries:
            out_df = pl.DataFrame(summaries)
            out_df.write_csv(output_csv_path)
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

    # ----------------------------
    # Haversine distance (meters)
    # ----------------------------
    @staticmethod
    def __haversine_distance_m(
        point1: tuple[float, float], point2: tuple[float, float]
    ) -> float:
        return haversine(point1, point2, unit=Unit.METERS, check=True, normalize=True)
