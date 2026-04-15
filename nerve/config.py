"""
Central configuration for the NERVE package.

Data root resolution priority:
  1. Explicit argument passed to functions / --data-root on CLI
  2. NERVE_DATA_ROOT environment variable
  3. Default: ~/.nerve/data/
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

ARTICLE_UUID = "c7edd728-fc40-4890-88f6-a71b171851dd"
PRIVATE_LINK = "BaEVPhT4moLWOb77YHjr8lzQlaF2F1vG441wRd3i7ek"
BASE_URL = "https://data.4tu.nl"
FILE_URL_TEMPLATE = f"{BASE_URL}/file/{ARTICLE_UUID}/{{file_uuid}}"
REFERER = f"{BASE_URL}/private_datasets/{PRIVATE_LINK}"
DATASET_PAGE_URL = f"{BASE_URL}/private_datasets/{PRIVATE_LINK}"

DEFAULT_DATA_ROOT = Path.home() / ".nerve" / "data"


def get_data_root(override: Optional[Union[str, Path]] = None) -> Path:
    """Resolve the data root directory.

    Args:
        override: Explicit path. Takes highest priority.

    Returns:
        Resolved Path to the data root directory.
    """
    if override is not None:
        return Path(override).expanduser().resolve()

    env = os.environ.get("NERVE_DATA_ROOT")
    if env:
        return Path(env).expanduser().resolve()

    return DEFAULT_DATA_ROOT.expanduser().resolve()
