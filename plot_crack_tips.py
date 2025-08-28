import os

from crackmnist import CrackMNIST

import matplotlib.pyplot as plt
import numpy as np



def plot_crack_tips(dataset, pixels, save_name="crack_tips_density.png"):
    crack_tips_summed = np.zeros((pixels, pixels))

    for i in range(len(dataset)):
        _, mask = dataset[i]
        crack_tips_summed += mask

    plt.imshow(crack_tips_summed, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Crack Tip Density')
    plt.title('Crack Tip Density Across Dataset')
    plt.xlabel('Pixel X')
    plt.ylabel('Pixel Y')
    plt.savefig(save_name)
    plt.close()


if __name__ == "__main__":
    if not os.path.exists(os.path.join("examples", "crack_tip_density")):
        os.makedirs(os.path.join("examples", "crack_tip_density"))

    for size in ["S", "M", "L"]:
        for pixels in [28]: #, 64, 128]:
            for split in ["train", "val", "test"]:
                print(f"Processing {split} split for size {size} and pixels {pixels}")
                dataset = CrackMNIST(split=split, pixels=pixels, size=size)
                plot_crack_tips(dataset, pixels=pixels,
                                save_name=os.path.join("examples", "crack_tip_density",
                                                       f"crack_tips_{split}_{size}_{pixels}.png"))
