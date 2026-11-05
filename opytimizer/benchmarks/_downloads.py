"""On-demand download of the official data of the CEC benchmark suites.

The shift vectors, rotation matrices and shuffles of the larger suites are not
shipped with the package; they are fetched from the organizers' repository the
first time a problem needs them and stored inside the package.
"""

import hashlib
import io
import os
import re
import shutil
import tempfile
import threading
import urllib.request
import zipfile
from pathlib import Path

import opytimizer.utils.exception as e
from opytimizer.utils import logging

logger = logging.get_logger(__name__)

_DATA_DIR = Path(__file__).resolve().parent / "data"

# Organizers' archive (code and input data) and its expected SHA-256 digest
_CEC2017_URL = (
    "https://raw.githubusercontent.com/P-N-Suganthan/CEC2017-BoundContrained/"
    "master/CEC17_fast_pow-C++.zip"
)
_CEC2017_SHA256 = "51e7f55dc5cddf263ec128771dee1db6a2d0e2816b3d650160eb51960f269a1e"

# Only these files are extracted from the archive (by their base name)
_CEC2017_FILES = re.compile(r"(shift_data_\d+|M_\d+_D\d+|shuffle_data_\d+_D\d+)\.txt")

_LOCK = threading.Lock()


def _download(url: str, sha256) -> bytes:
    """Downloads `url` and checks its SHA-256 digest (when one is pinned)."""
    logger.info("Downloading %s.", url)
    request = urllib.request.Request(url, headers={"User-Agent": "opytimizer"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()

    digest = hashlib.sha256(payload).hexdigest()
    if sha256 is not None and digest != sha256:
        raise e.ValueError(
            f"Checksum mismatch for {url} (expected {sha256}, got {digest})"
        )

    return payload


def _extract(payload: bytes, pattern, target: Path) -> int:
    """Extracts the members of a zip whose base name matches `pattern` into `target`.

    Files are written to a temporary folder first and then moved, so a failed or
    interrupted extraction never leaves a partial data folder behind.
    """

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        count = 0
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            for member in archive.infolist():
                # Only the base name is kept, so no path in the archive is trusted
                name = os.path.basename(member.filename.replace("\\", "/"))
                if member.is_dir() or not pattern.fullmatch(name):
                    continue
                with archive.open(member) as source, open(staging / name, "wb") as sink:
                    shutil.copyfileobj(source, sink)
                count += 1

        if count == 0:
            raise e.ValueError("The downloaded archive contains no data files")

        if target.exists():
            shutil.rmtree(target)
        staging.rename(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)

    return count


def cec2017_data_dir() -> Path:
    """Returns the folder with the CEC 2017 data, downloading it if needed.

    Returns:
        (Path): Folder holding `shift_data_*`, `M_*_D*` and `shuffle_data_*` files.

    """

    target = _DATA_DIR / "cec2017"

    with _LOCK:
        if target.is_dir() and any(target.glob("shift_data_*.txt")):
            return target

        try:
            payload = _download(_CEC2017_URL, _CEC2017_SHA256)
            count = _extract(payload, _CEC2017_FILES, target)
        except (OSError, zipfile.BadZipFile) as error:
            raise e.Error(
                "DataError",
                f"Could not obtain the CEC 2017 data ({error}). Download {_CEC2017_URL}, "
                f"and extract its `input_data` files into {target}.",
            ) from error

        logger.info("Stored %d CEC 2017 data files in %s.", count, target)

    return target
