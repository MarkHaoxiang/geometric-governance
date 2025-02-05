import os

import gdown

from crypto_l65.utils import DATA_DIR, info

URL = "https://drive.google.com/drive/folders/1MRPXz79Lu_JGLlJ21MDfML44dKN9R08l"


def _download_dataset_if_not_exists():
    dir = os.path.join(DATA_DIR, "raw")
    if not os.path.exists(dir):
        info("Downloading Elliptic++ dataset")
        os.makedirs(dir)
        gdown.download_folder(url=URL, output=dir)


_download_dataset_if_not_exists()
