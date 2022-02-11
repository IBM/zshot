import functools
import os
import pathlib
import shutil
from urllib.request import urlopen
import requests
from tqdm.auto import tqdm


def download_file(url, output_dir="."):
    filename = url.rsplit('/', 1)[1]
    path = pathlib.Path(os.path.join(output_dir, filename)).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True) as r:
        total_length = int(urlopen(url=url).info().get('Content-Length', 0))
        if path.exists() and os.path.getsize(path) == total_length:
            return
        r.raw.read = functools.partial(r.raw.read, decode_content=True)
        with tqdm.wrapattr(r.raw, "read", total=total_length, desc=f"Downloading {filename}") as raw:
            with path.open("wb") as output:
                shutil.copyfileobj(raw, output)