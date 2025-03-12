import os
import h5py
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
from crackmnist.info import DEFAULT_ROOT, INFO, HOMEPAGE


class CrackMNIST(Dataset):
    flag = "crackmnist"
    available_splits = ["train", "val", "test"]

    def __init__(
        self,
        split,
        transform=None,
        target_transform=None,
        size: str = "S",
        pixels: int = 28,
        download_path: str = DEFAULT_ROOT
    ):
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.size = size
        self.pixels = pixels

        self.info = INFO[self.flag]

        self.available_pixels = [28, 64, 128, 256]
        self.available_sizes = ["S", "M", "L"]
        assert size in self.available_sizes
        assert pixels in self.available_pixels

        assert (
            self.split in self.available_splits
        ), f"Split {split} is not available, use one of {self.available_splits}!"

        self.download_path = download_path
        self.download()

        if not os.path.exists(
            os.path.join(self.download_path, f"{self.flag}_{self.pixels}_{self.size}.h5")
        ):
            raise RuntimeError("Dataset not found.")

        hf = h5py.File(os.path.join(self.download_path, f"{self.flag}_{self.pixels}_{self.size}.h5"))
        self.images = hf[f"{self.split}_images"]
        self.masks = hf[f"{self.split}_masks"]

    def download(self):
        if os.path.exists(
            os.path.join(self.download_path, f"{self.flag}_{self.pixels}_{self.size}.h5")
        ):
            return

        if self.pixels == 256 and self.size in ["M", "L"]:
            raise RuntimeError(
                f"{self.flag}_{self.pixels}_{self.size}.h5 is not available on Zenodo. "
                f"Please contact the authors to get access."
            )

        try:
            download_url(
                url=self.info[f"url_{self.pixels}_{self.size}"],
                root=self.root,
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
        if isinstance(idx, int):
            img, target = self.images[idx], self.masks[idx]

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

            return img, target

        img, target = self.images[idx], self.masks[idx]

        if self.transform is not None:
            img = [self.transform(x) for x in img]

        if self.target_transform is not None:
            target = [self.target_transform(x) for x in target]

        return img, target
