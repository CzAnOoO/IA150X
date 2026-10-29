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
original_output_dir = "original_dataset"

label_map = {1: "meningioma", 2: "glioma", 3: "pituitary"}

for split in ["train", "val", "test"]:
    for label in ["meningioma", "glioma", "pituitary"]:
        os.makedirs(f"{output_dir}/{split}/{label}", exist_ok=True)
        os.makedirs(f"{original_output_dir}/{split}/{label}", exist_ok=True)


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

    # some images of brain tumors is larger than 224 pixel (e.g. 2132,2763 ...)
    if h > 224 or w > 224:
        scale = 224 / max(h, w)
        new_h = int(h * scale)
        new_w = int(w * scale)

        cropped = Image.fromarray(cropped)
        cropped = cropped.resize((new_w, new_h), Image.BILINEAR)
        cropped = np.array(cropped)

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

    return padded, img, label


import random

for folder in input_dirs:
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        try:
            copped_img, img, label = process_mat(path)

            # random
            r = random.random()
            if r < 0.8:
                split = "train"
            elif r < 0.9:
                split = "val"
            else:
                split = "test"

            temp_img = Image.fromarray(img)
            temp_img = temp_img.resize((224, 224), Image.BILINEAR)
            img = np.array(temp_img)

            # intensity normalization: Max-Min
            copped_img = (
                (copped_img - copped_img.min())
                / (copped_img.max() - copped_img.min())
                * 255
            ).astype(np.uint8)
            img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)

            # save as .png
            copped_im = Image.fromarray(copped_img)
            save_path = f"{output_dir}/{split}/{label_map[label]}/{file}.png"
            copped_im.save(save_path)

            im = Image.fromarray(img)
            save_path = f"{original_output_dir}/{split}/{label_map[label]}/{file}.png"
            im.save(save_path)

        except Exception as e:
            print(f"Error processing file: {path} | Exception: {e}")
