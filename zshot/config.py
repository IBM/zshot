import os
import pathlib

MODELS_CACHE_PATH = os.getenv("MODELS_CACHE_PATH") if "MODELS_CACHE_PATH" in os.environ \
    else f"{pathlib.Path.home()}/.cache/zshot/"
