import datetime
import gc
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
            .str.strptime(pl.Date, "%Y-%m-%d", strict=False)
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
        self, output_csv_path: str = f"mesa_initializers.csv"
    ):
        output_csv_path = self.__data_path / output_csv_path
        max_time = self.date_time + datetime.timedelta(hours=1)
        min_time = self.date_time - datetime.timedelta(hours=1)

        trips_df = self.__reading_trips_df_and_gathering_their_data
        chosen_trips = trips_df.select(
            ["ID", "Main_Mode", "start_datetime", "end_datetime"]
        ).to_dicts()
        del trips_df
        gc.collect()

        summaries: List[Dict[str, Any]] = []

        for trip_record in chosen_trips:
            agent_id = trip_record.get("ID")
            main_mode = trip_record.get("Main_Mode")
            if not agent_id:
                continue

            gps_path = f"{self.__data_path}/gps_dataset/{agent_id}.csv"
            try:
                df_c = pl.read_csv(gps_path, try_parse_dates=False)
            except FileNotFoundError:
                print(f"Warning: GPS file not found for ID {agent_id}")
                continue

            # parse GPS local datetimes
            df_c = df_c.with_columns(
                pl.col("LOCAL DATETIME")
                .str.strptime(pl.Datetime, "%Y-%m-%d-%H-%M-%S", strict=False)
                .alias("local_dt")
            )

            # windowed GPS
            df_win = df_c.filter(pl.col("local_dt").is_between(min_time, max_time))
            if df_win.is_empty():
                continue

            # lists for quick access
            lats = df_win["LATITUDE"].to_list()
            lons = df_win["LONGITUDE"].to_list()
            times = df_win["local_dt"].to_list()
            speeds_raw = (
                df_win.get_column("SPEED").to_list()
                if "SPEED" in df_win.columns
                else [None] * len(lats)
            )

            # clean speeds and compute stats
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

            # find sample closest to center
            min_d = float("inf")
            best_idx = None
            for idx, (lat, lon) in enumerate(zip(lats, lons)):
                if lat is None or lon is None:
                    continue
                # call your haversine with scalar args
                d = self.__haversine_distance_m((lat, lon), self.center)
                if d < min_d:
                    min_d = d
                    best_idx = idx

            if best_idx is None:
                continue

            # choose starting point (closest sample)
            start_lat = lats[best_idx]
            start_lon = lons[best_idx]
            start_time = times[best_idx]
            inferred_mode = None
            for trip in chosen_trips:
                if trip.get("start_datetime") <= start_time <= trip.get("end_datetime"):
                    inferred_mode = trip.get("Main_Mode")
                    break

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
            }
            summaries.append(summary)

        # write output CSV
        if summaries:
            out_df = pl.DataFrame(summaries)
            out_df.write_csv(output_csv_path)

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
