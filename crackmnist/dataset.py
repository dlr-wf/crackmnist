import os
import h5py
import json

from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
from crackmnist.info import DEFAULT_ROOT, INFO, HOMEPAGE


class CrackMNIST(Dataset):
    flag = "crackmnist"
    available_splits = ["train", "val", "test"]
    available_sizes = ["S", "M", "L"]
    available_pixels = [28, 64, 128, 256]
    available_tasks = ["crack_tip_segmentation", "SIF_regression"]

    def __init__(
        self,
        split,
        transform=None,
        target_transform=None,
        size: str = "S",
        pixels: int = 28,
        task: str = "crack_tip_segmentation",
        download_path: str = DEFAULT_ROOT
    ):
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.size = size
        self.pixels = pixels
        self.task = task
        self.info = INFO[self.flag]

        assert size in self.available_sizes, f"Size {size} is not available, use one of {self.available_sizes}!"
        assert isinstance(pixels, int), f"Pixels must be an integer, got {type(pixels)}!"
        assert pixels in self.available_pixels, f"Pixels {pixels} are not available, use one of {self.available_pixels}!"
        assert task in self.available_tasks, f"Task {task} is not available, use one of {self.available_tasks}!"
        assert (
            self.split in self.available_splits
        ), f"Split {split} is not available, use one of {self.available_splits}!"

        self.download_path = download_path
        self.download()

        if not os.path.exists(
            os.path.join(self.download_path, f"{self.flag}_{self.pixels}_{self.size}.h5")
        ):
            raise RuntimeError("Dataset not found.")

        # Description
        if self.task == "crack_tip_segmentation":
            self.description = (
                "CrackMNIST is a dataset of 2-channel DIC images (u_x, u_y displacements) as inputs from fatigue "
                "crack growth experiments and corresponding crack tip segmentation masks as targets."
            )
        if self.task == "SIF_regression":
            self.description = (
                "CrackMNIST is a dataset of 2-channel DIC images (u_x, u_y displacements) as inputs from fatigue "
                "crack growth experiments with corresponding stress intensity factors (SIFs) (K_I, K_II, T-Stress) as targets."
            )

        # Load data
        hf = h5py.File(os.path.join(self.download_path, f"{self.flag}_{self.pixels}_{self.size}.h5"))
        self.images = hf[f"{self.split}_images"]
        self.exp_ids = hf[f"{self.split}_exp_ids"]
        self.exp_names = hf["experiments"].asstr()
        if self.task == "crack_tip_segmentation":
            self.targets = hf[f"{self.split}_masks"]
        if self.task == "SIF_regression":
            self.targets = hf[f"{self.split}_SIFs"]
        self.forces = hf[f"{self.split}_forces"]
        self.augmentations = hf[f"{self.split}_augs"]

        # Load metadata for experiments
        with open(os.path.join(self.download_path, "experiments_metadata.json"), "r") as f:
            self.metadata = json.load(f)


    def download(self):
        if self.pixels == 256 and self.size in ["M", "L"]:
            raise RuntimeError(
                f"{self.flag}_{self.pixels}_{self.size}.h5 is not available on Zenodo. "
                f"Please contact the authors to get access."
            )

        if not os.path.exists(os.path.join(self.download_path, "experiments_metadata.json")):
            try:
                download_url(
                    url=self.info["metadata_url"],
                    root=self.download_path,
                    filename="experiments_metadata.json",
                    md5=self.info["MD5_metadata"],
                )
            except Exception as e:
                raise RuntimeError(
                    f"""
                    Automatic download of metadata failed! Please download experiments_metadata.json manually.
                    1. [Optional] Check your network connection: 
                        Go to {HOMEPAGE} and find the Zenodo repository
                    2. Download the metadata file from the Zenodo repository or its Zenodo data link: 
                        {self.info["metadata_url"]}
                    3. [Optional] Verify the MD5: 
                        {self.info["MD5_metadata"]}
                    4. Put the metadata file under your CrackMNIST root folder: 
                        {self.download_path}
                    """
                ) from e

        if not os.path.exists(os.path.join(self.download_path, f"{self.flag}_{self.pixels}_{self.size}.h5")):
            try:
                download_url(
                    url=self.info[f"url_{self.pixels}_{self.size}"],
                    root=self.download_path,
                    filename=f"{self.flag}_{self.pixels}_{self.size}.h5",
                    md5=self.info[f"MD5_{self.pixels}_{self.size}"],
                )
            except Exception as e:
                raise RuntimeError(
                    f"""
                    Automatic download failed! Please download {self.flag}_{self.pixels}_{self.size}.h5 manually.
                    1. [Optional] Check your network connection: 
                        Go to {HOMEPAGE} and find the Zenodo repository
                    2. Download the h5-file from the Zenodo repository or its Zenodo data link: 
                        {self.info[f"url_{self.pixels}_{self.size}"]}
                    3. [Optional] Verify the MD5: 
                        {self.info[f"MD5_{self.pixels}_{self.size}"]}
                    4. Put the h5-file under your CrackMNIST root folder: 
                        {self.download_path}
                    """
                ) from e

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        """
        return:
            img: np.ndarray
            target: np.ndarray
        """
        img, target = self.images[idx], self.targets[idx]

        if isinstance(idx, int):
            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

        else:
            if self.transform is not None:
                img = [self.transform(x) for x in img]

            if self.target_transform is not None:
                target = [self.target_transform(x) for x in target]

        return img, target

    def get_metadata(self, idx):
        """Get metadata for the sample at index idx.

        :param idx: int or list of int or numpy array of int
        :return: single dict or list of dicts with keys "exp_id" and "experiment_name"
        """

        if isinstance(idx, int):
            name = self.exp_names[int(self.exp_ids[idx])]
            return self.metadata[name]

        names = [self.exp_names[int(i)] for i in self.exp_ids[idx]]
        metas = [self.metadata[name] for name in names]

        return metas

    def get_forces(self, idx):
        """Get forces for the sample at index idx.

        :param idx: int or list of int or numpy array of int
        :return: single float or list of floats
        """

        if isinstance(idx, int):
            return float(self.forces[idx])

        return [float(f) for f in self.forces[idx]]

    def get_augmentations(self, idx):
        """Get augmentations for the sample at index idx.
        :param idx: int or list of int or numpy array of int
        :return: single dict or list of dicts with keys "shift", "rotation", and "vertical_flip"
        0: shift_x in mm, 1: shift_y in mm, 2: rotation in deg, 3: vertical_flip as bool
        """

        if isinstance(idx, int):
            return {"shift": (float(self.augmentations[idx][0]), float(self.augmentations[idx][1])),
                    "rotation": float(self.augmentations[idx][2]),
                    "vertical_flip": bool(self.augmentations[idx][3])}

        return [{"shift": (float(self.augmentations[i][0]), float(self.augmentations[i][1])),
                 "rotation": float(self.augmentations[i][2]),
                 "vertical_flip": bool(self.augmentations[i][3])} for i in idx]
