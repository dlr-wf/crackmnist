import os
import warnings

__version__ = "0.0.1"


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
        "description": "Digital image correlation data of fatigue crack growth experiments",
        "url_28_S": "https://zenodo.org/records/-/files/crackmnist_28_S.h5?download=1",
        "url_64_S": "https://zenodo.org/records/-/files/crackmnist_64_S.h5?download=1",
        "url_128_S": "https://zenodo.org/records/-/files/crackmnist_128_S.h5?download=1",
        "url_256_S": "https://zenodo.org/records/-/files/crackmnist_256_S.h5?download=1",
        "url_28_M": "https://zenodo.org/records/-/files/crackmnist_28_M.h5?download=1",
        "url_64_M": "https://zenodo.org/records/-/files/crackmnist_64_M.h5?download=1",
        "url_128_M": "https://zenodo.org/records/-/files/crackmnist_128_M.h5?download=1",
        "url_256_M": "https://zenodo.org/records/-/files/crackmnist_256_M.h5?download=1",
        "url_28_L": "https://zenodo.org/records/-/files/crackmnist_28_L.h5?download=1",
        "url_64_L": "https://zenodo.org/records/-/files/crackmnist_64_L.h5?download=1",
        "url_128_L": "https://zenodo.org/records/-/files/crackmnist_128_L.h5?download=1",
        "url_256_L": "https://zenodo.org/records/-/files/crackmnist_256_L.h5?download=1",

        "MD5_28_S": "-",
        "MD5_64_S": "-",
        "MD5_128_S": "-",
        "MD5_256_S": "-",
        "MD5_28_M": "-",
        "MD5_64_M": "-",
        "MD5_128_M": "-",
        "MD5_256_M": "-",
        "MD5_28_L": "-",
        "MD5_64_L": "-",
        "MD5_128_L": "-",
        "MD5_256_L": "-",

        "task": "semantic segmentation",
        "label": {"crack_tip": 1, "no_crack_tip": 0},
        "n_channels": 2,
        "n_samples": {
            "train": {"S": 10048, "M": 21672, "L": 42088},
            "val": {"S": 5944, "M": 11736, "L": 11736},
            "test": {"S": 5944, "M": 11672, "L": 16560},
        },
        "license": "",
    }
}

HOMEPAGE = "https://github.com/dlr-wf/CrackMNIST/"
