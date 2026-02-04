import os
import warnings

__version__ = "2.0.0"


def get_default_root():
    home = os.path.expanduser("~")
    dirpath = os.path.join(home, ".crackmnist")

    try:
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
    except Exception as e:
        warnings.warn(f"Failed to setup default root. {e}")
        dirpath = None

    return dirpath


DEFAULT_ROOT = get_default_root()

INFO = {
    "crackmnist": {
        "python_class": "CrackMNIST",
        "description": "Annotated digital image correlation displacement fields from fatigue crack growth experiments",

        "metadata_url": "https://zenodo.org/records/18454958/files/experiments_metadata.json?download=1",

        "url_28_S": "https://zenodo.org/records/18454958/files/crackmnist_28_S.h5?download=1",
        "url_64_S": "https://zenodo.org/records/18454958/files/crackmnist_64_S.h5?download=1",
        "url_128_S": "https://zenodo.org/records/18454958/files/crackmnist_128_S.h5?download=1",

        "url_28_M": "https://zenodo.org/records/18454958/files/crackmnist_28_M.h5?download=1",
        "url_64_M": "https://zenodo.org/records/18454958/files/crackmnist_64_M.h5?download=1",
        "url_128_M": "https://zenodo.org/records/18454958/files/crackmnist_128_M.h5?download=1",

        "url_28_L": "https://zenodo.org/records/18454958/files/crackmnist_28_L.h5?download=1",
        "url_64_L": "https://zenodo.org/records/18454958/files/crackmnist_64_L.h5?download=1",
        "url_128_L": "https://zenodo.org/records/18454958/files/crackmnist_128_L.h5?download=1",

        "MD5_metadata": "85b558aa217c2ad659b701946c399f58",

        "MD5_28_S": "26bb0aa814f2e3ed467879844222c46c",
        "MD5_64_S": "948d0ba363b5d06a904d6cdcedc029e8",
        "MD5_128_S": "3101a618e0837276b1ef4533964fabb3",

        "MD5_28_M": "18df30fbee389def264ebfd25377e6d2",
        "MD5_64_M": "1c4c40674d5781dc42d20627b7b460b7",
        "MD5_128_M": "53518c6684eb006527fdf75e4efb1f5c",

        "MD5_28_L": "b031a047c61bfd14ddd294a63afc94ac",
        "MD5_64_L": "5963d788af31f4a24ebe904cb3ad43db",
        "MD5_128_L": "5527a54cc382623edcb2022e95a9ed2d",

        "n_channels": 2,
        "n_samples": {
            "train": {"S": 10048, "M": 21672, "L": 42088},
            "val": {"S": 5944, "M": 11736, "L": 11736},
            "test": {"S": 5944, "M": 11672, "L": 16560},
        },
        "license": "CC BY 4.0",
    }
}

HOMEPAGE = "https://github.com/dlr-wf/crackmnist/"
