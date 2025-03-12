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
        "url_28_S": "https://zenodo.org/records/15013128/files/crackmnist_28_S.h5?download=1",
        "url_64_S": "https://zenodo.org/records/15013128/files/crackmnist_64_S.h5?download=1",
        "url_128_S": "https://zenodo.org/records/15013128/files/crackmnist_128_S.h5?download=1",
        "url_256_S": "https://zenodo.org/records/15013128/files/crackmnist_256_S.h5?download=1",
        "url_28_M": "https://zenodo.org/records/15013128/files/crackmnist_28_M.h5?download=1",
        "url_64_M": "https://zenodo.org/records/15013128/files/crackmnist_64_M.h5?download=1",
        "url_128_M": "https://zenodo.org/records/15013128/files/crackmnist_128_M.h5?download=1",
        "url_256_M": "-",
        "url_28_L": "https://zenodo.org/records/15013128/files/crackmnist_28_L.h5?download=1",
        "url_64_L": "https://zenodo.org/records/15013128/files/crackmnist_64_L.h5?download=1",
        "url_128_L": "https://zenodo.org/records/15013128/files/crackmnist_128_L.h5?download=1",
        "url_256_L": "-",

        "MD5_28_S": "6db8fcb85274889f18af406cc7acead7",
        "MD5_64_S": "a792fd1696b3f2d9d42ac52cec70f6b9",
        "MD5_128_S": "01957bb66136725bf9d5dd4ecfbd067b",
        "MD5_256_S": "579f127cd296638ccb15f695a8ed3e48",
        "MD5_28_M": "58466521aa08dcc65a84d224e821985f",
        "MD5_64_M": "7302c0b37f04d72e6dc4b953b7bd3e81",
        "MD5_128_M": "e10197e36aa26eb51981ab92c6dea905",
        "MD5_256_M": "-",
        "MD5_28_L": "9e1ff7ff395b2dbeb250e0484c78d3a1",
        "MD5_64_L": "a186337e3b8fd3b9aeee86b0e3c30c18 ",
        "MD5_128_L": "f4568b3d80afc93eadac2524c20f67d3",
        "MD5_256_L": "-",

        "task": "semantic segmentation",
        "label": {"crack_tip": 1, "no_crack_tip": 0},
        "n_channels": 2,
        "n_samples": {
            "train": {"S": 10048, "M": 21672, "L": 42088},
            "val": {"S": 5944, "M": 11736, "L": 11736},
            "test": {"S": 5944, "M": 11672, "L": 16560},
        },
        "license": "CC BY 4.0",
    }
}

HOMEPAGE = "https://github.com/dlr-wf/CrackMNIST/"
