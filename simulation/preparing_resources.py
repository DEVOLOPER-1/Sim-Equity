import requests
import datetime
from alive_progress import alive_bar
from pathlib import Path


class Logger:
    def __init__(self, file_path: Path):
        self.FILE_PATH = file_path
        # ensure the parent directory exists
        self.FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

    def log_it(self, file_name: str):
        timestamp = datetime.datetime.now().isoformat( timespec="seconds")
        entry = (
            f"FILE NAME: {file_name} | "
            f"TIME UPDATED: {timestamp} | "
            f"FORMAT: {Path(file_name).suffix.lstrip('.')}\n"
        )
        # append rather than overwrite
        with self.FILE_PATH.open(mode="a", encoding="utf-8") as f:
            f.write(entry)


class Downloader:
    # anchor data dir next to this script, not relative to CWD
    DATA_PATH = Path(__file__).parent / "data"
    LOG_PATH  = DATA_PATH / "downloads.log"

    def __init__(self):
        # make sure the data directory exists
        self.DATA_PATH.mkdir(parents=True, exist_ok=True)
        self.logger = Logger(self.LOG_PATH)

    def _ping_it(self, url: str) -> bool:
        try:
            return requests.get(url, timeout=10).status_code == 200
        except requests.RequestException:
            return False

    def _content_catcher(self, url: str) -> requests.Response:
        return requests.get(url, stream=True, allow_redirects=True, timeout=30)

    def _download_file_and_log_it(self, url: str, filename: str):
        if not self._ping_it(url):
            print(f"❌ Failed to download {filename}: URL not accessible")
            return

        response = self._content_catcher(url)
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

    def rail_network(self):
        url = (
            "https://data.iledefrance-mobilites.fr/api/explore/v2.1/"
            "catalog/datasets/emplacement-des-gares-idf/exports/"
            "parquet?lang=fr&timezone=Asia%2FBeirut"
        )
        self._download_file_and_log_it(url, "emplacement-des-gares-idf.parquet")
