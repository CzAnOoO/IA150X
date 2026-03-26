# import scipy.io as sio
import h5py
import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage
from PIL import Image, ImageOps

""" 
https://docs.h5py.org/en/3.5.0/quick.html
 """
# Error processing file: data/brainTumorDataPublic_15332298/2132.mat | Exception: index can't contain negative values
# Error processing file: data/brainTumorDataPublic_22993064/2763.mat | Exception: index can't contain negative values
# Error processing file: data/brainTumorDataPublic_22993064/2762.mat | Exception: index can't contain negative values
# Error processing file: data/brainTumorDataPublic_22993064/2370.mat | Exception: index can't contain negative values
# Error processing file: data/brainTumorDataPublic_22993064/2371.mat | Exception: index can't contain negative values
# Error processing file: data/brainTumorDataPublic_22993064/2761.mat | Exception: index can't contain negative values
# Error processing file: data/brainTumorDataPublic_22993064/2759.mat | Exception: index can't contain negative values
# Error processing file: data/brainTumorDataPublic_22993064/2939.mat | Exception: index can't contain negative values
# Error processing file: data/brainTumorDataPublic_22993064/2379.mat | Exception: index can't contain negative values
# Error processing file: data/brainTumorDataPublic_22993064/2768.mat | Exception: index can't contain negative values
# Error processing file: data/brainTumorDataPublic_22993064/2382.mat | Exception: index can't contain negative values
# Error processing file: data/brainTumorDataPublic_22993064/2989.mat | Exception: index can't contain negative values
# Error processing file: data/brainTumorDataPublic_22993064/2380.mat | Exception: index can't contain negative values
# Error processing file: data/brainTumorDataPublic_22993064/2381.mat | Exception: index can't contain negative values
with h5py.File("data/brainTumorDataPublic_22993064/2382.mat", "r") as f:
    # Read .mat files in Python: https://codemia.io/knowledge-hub/path/read_mat_files_in_python
    print(list(f.keys()))

    cjdata = f["cjdata"]
    print(list(cjdata.keys()))

    # https://medium.com/@gopiprasanthpotipireddy/using-mat-files-in-python-77293f995a43
    img = np.array(f["/cjdata/image"])
    PID = np.array(f["/cjdata/PID"])
    label = np.array(f["/cjdata/label"]).item()
    tumorBorder = np.array(f["/cjdata/tumorBorder"])
    tumorMask = np.array(f["/cjdata/tumorMask"])

    img = img.T
    tumorMask = tumorMask.T
    iMg = img * tumorMask

    # np.savetxt("image_matrix.txt", img, fmt="%f")
    print(f"Label {label}")
    plt.subplot(2, 3, 1)
    plt.title(f"Image label={label}")
    plt.imshow(img, cmap="gray")

    plt.subplot(2, 3, 2)
    plt.title("Mask")
    plt.imshow(tumorMask, cmap="gray")

    plt.subplot(2, 3, 3)
    plt.title("Image && Mask")
    plt.imshow(iMg, cmap="gray")

    # Cropping: https://www.askpython.com/python-modules/scipy/scipy-ndimage
    labeled_array, num_features = ndimage.label(iMg > 0)

    slice = ndimage.find_objects(labeled_array)
    tumor = slice[0]
    iMg_cropped = iMg[tumor]

    plt.subplot(2, 3, 4)
    plt.title("Cropping")
    plt.imshow(iMg_cropped, cmap="gray")

    # Padding
    current_h, current_w = iMg_cropped.shape

    if current_h > 224 or current_w > 224:
        scale = 224 / max(current_h, current_w)
        new_h = int(current_h * scale)
        new_w = int(current_w * scale)

        iMg_cropped = Image.fromarray(iMg_cropped)
        iMg_cropped = iMg_cropped.resize((new_w, new_h), Image.BILINEAR)
        iMg_cropped = np.array(iMg_cropped)

        current_h, current_w = iMg_cropped.shape

    # Goal is 224 x 224 pixel
    pad_h = 224 - current_h
    pad_w = 224 - current_w

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    iMg_padded = np.pad(
        iMg_cropped, pad_width=((pad_top, pad_bottom), (pad_left, pad_right))
    )

    plt.subplot(2, 3, 5)
    plt.title("Padding: 224 x 224")
    plt.imshow(iMg_padded, cmap="gray")

    plt.show()
