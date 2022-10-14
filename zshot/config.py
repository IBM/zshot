import os
import pathlib

from appdata import AppDataPaths

MODELS_CACHE_PATH = os.getenv("MODELS_CACHE_PATH") if "MODELS_CACHE_PATH" in os.environ \
    else AppDataPaths(f"{pathlib.Path(__file__).stem}").app_data_path + "/"
