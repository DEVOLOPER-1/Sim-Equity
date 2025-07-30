import logging

import requests
import datetime
from alive_progress import alive_bar
from pathlib import Path
from zipfile import ZipFile
import osmnx as ox
import os


class Logger:
    def __init__(self, file_path: Path):
        self.FILE_PATH = file_path
        # ensure the parent directory exists
        self.FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

    def log_it(self, file_name: str)->None:
        timestamp = datetime.datetime.now().isoformat(timespec="seconds")
        entry = (
            f"FILE NAME: {file_name} | "
            f"TIME UPDATED: {timestamp} | "
            f"FORMAT: {Path(file_name).suffix.lstrip('.')}\n"
        )
        # append rather than overwrite
        with self.FILE_PATH.open(mode="a", encoding="utf-8") as f:
            f.write(entry)


class Downloader:
    """
    Does the main logic of downloading & requests & saving the files with decompression capabilities if needed
    """
    # anchor data dir next to this script, not relative to CWD
    DATA_PATH = Path(__file__).parent / "data"
    LOG_PATH = DATA_PATH / "downloads.log"

    def __init__(self):
        # make sure the data directory exists
        self.DATA_PATH.mkdir(parents=True, exist_ok=True)
        self.logger = Logger(self.LOG_PATH)

    @staticmethod
    def __ping_it(url: str) -> bool:
        try:
            return requests.get(url, timeout=10).status_code == 200
        except requests.RequestException:
            return False

    @staticmethod
    def __content_catcher(url: str) -> requests.Response:
        return requests.get(url, stream=True, allow_redirects=True, timeout=30)

    @staticmethod
    def __identify_is_compressed_then_extract_then_delete_source(filename: str, target_path: str) -> None:
        if filename.endswith("zip"):
            with ZipFile(target_path, "r") as myzip:
                myzip.extractall(target_path.replace(".zip", ""))

            os.remove(target_path)
        else:
            return

    def download_file_and_log_it(self, url: str, filename: str) -> None:
        if not self.__ping_it(url):
            print(f"❌ Failed to download {filename}: URL not accessible")
            return

        response = self.__content_catcher(url)
        total_size = int(response.headers.get("content-length", 0))

        target_path = self.DATA_PATH / filename
        with alive_bar(total=total_size, title=filename) as bar:
            # stream in chunks so that the progress bar can actually update
            with target_path.open("wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    if not chunk:
                        continue
                    file.write(chunk)
                    bar(len(chunk))

        # only once the file is fully written do we log
        self.logger.log_it(filename)
        self.__identify_is_compressed_then_extract_then_delete_source(filename, str(target_path))


class IleDeFranceMobilityDataCollector(Downloader):
    """
    Downloads and processes Île-de-France transportation datasets.

    This class extends the base Downloader to fetch specific mobility and transit
    data for the Paris metropolitan area, including:
    - Rail network station locations and metadata
    - GTFS (General Transit Feed Specification) data for urban/interurban networks
    - Real-time transit information and schedules

    All downloaded files are automatically logged and compressed archives are
    extracted to the configured data directory.
    """


    def __init__(self):
        super().__init__()
        ...

    def stations_of_ile_de_france_rail_network(self):
        url = (
            "https://data.iledefrance-mobilites.fr/api/explore/v2.1/"
            "catalog/datasets/emplacement-des-gares-idf/exports/"
            "parquet?lang=en&timezone=Africa%2FCairo"
        )
        self.download_file_and_log_it(url, "emplacement-des-gares-idf.parquet")

    def urban_and_interurban_network_of_ile_de_france_mobility(self):
        url = "https://www.data.gouv.fr/api/1/datasets/r/413988ed-d340-467b-8be2-7b999fcd207a"
        self.download_file_and_log_it(url, "idfm_gtfs.zip")

    def ile_de_france_open_street_map(self):
        url = "https://download.geofabrik.de/europe/france/ile-de-france-latest.osm.pbf"
        self.download_file_and_log_it(url, "ile-de-france-latest.osm.pbf")

    def ile_de_france_open_street_map_(self):
        place_name = "Île-de-France, France"
        G_drive = ox.graph_from_place(place_name, network_type='all', simplify=True, retain_all=False)
        ox.save_graphml(G_drive, filepath='./simulation/data/osmnx_layers/IDF_network.graphml')
        Logger.log_it("IDF_network.graphml")
