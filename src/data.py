import os
import h5py
import numpy as np
from scipy import ndimage
from PIL import Image

input_dirs = [
    "data/brainTumorDataPublic_1766",
    "data/brainTumorDataPublic_7671532",
    "data/brainTumorDataPublic_15332298",
    "data/brainTumorDataPublic_22993064",
]

output_dir = "processed_dataset"

label_map = {1: "meningioma", 2: "glioma", 3: "pituitary"}

for split in ["train", "val", "test"]:
    for label in ["meningioma", "glioma", "pituitary"]:
        os.makedirs(f"{output_dir}/{split}/{label}", exist_ok=True)


def process_mat(file_path):
    with h5py.File(file_path, "r") as f:
        img = np.array(f["/cjdata/image"]).T
        tumorMask = np.array(f["/cjdata/tumorMask"]).T
        label = int(np.array(f["/cjdata/label"]).item())

    roi = img * tumorMask

    # Cropping: https://www.askpython.com/python-modules/scipy/scipy-ndimage
    labeled_array, _ = ndimage.label(roi > 0)
    slices = ndimage.find_objects(labeled_array)
    tumor = slices[0]
    cropped = roi[tumor]

    # Padding
    h, w = cropped.shape
    pad_h = 224 - h
    pad_w = 224 - w
    # pad_top = pad_h // 2
    # pad_bottom = pad_h - pad_top
    # pad_left = pad_w // 2
    # pad_right = pad_w - pad_left
    padded = np.pad(
        cropped, ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2))
    )

    return padded, label


import random

for folder in input_dirs:
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        img, label = process_mat(path)

        # random
        r = random.random()
        if r < 0.8:
            split = "train"
        elif r < 0.9:
            split = "val"
        else:
            split = "test"

        # intensity normalization: Max-Min
        img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)

        # save as .png
        im = Image.fromarray(img)
        save_path = f"{output_dir}/{split}/{label_map[label]}/{file}.png"
        im.save(save_path)
