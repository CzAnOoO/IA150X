# import scipy.io as sio
import h5py
import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage
from PIL import Image, ImageOps

""" 
https://docs.h5py.org/en/3.5.0/quick.html
 """
with h5py.File("data/brainTumorDataPublic_1766/1.mat", "r") as f:
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
