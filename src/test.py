# import scipy.io as sio
import h5py
import numpy as np
import matplotlib.pyplot as plt

""" 
Read .mat files in Python: https://codemia.io/knowledge-hub/path/read_mat_files_in_python
https://docs.h5py.org/en/3.5.0/quick.html
 """
with h5py.File("data/brainTumorDataPublic_1766/1.mat", "r") as f:
    print(list(f.keys()))

    cjdata = f["cjdata"]
    print(list(cjdata.keys()))

    # img_ref = cjdata["image"]
    # mask_ref = cjdata["tumorMask"]
    # label_ref = cjdata["label"]

    # img = np.array(f[img_ref])
    # mask = np.array(f[mask_ref])
    # label = int(np.array(f[label_ref]))

    """ https://medium.com/@gopiprasanthpotipireddy/using-mat-files-in-python-77293f995a43 """
    img = np.array(f["/cjdata/image"])
    PID = np.array(f["/cjdata/PID"])
    label = int(np.array(f["/cjdata/label"]))
    tumorBorder = np.array(f["/cjdata/tumorBorder"])
    tumorMask = np.array(f["/cjdata/tumorMask"])

    img = img.T
    tumorMask = tumorMask.T

    plt.subplot(1, 2, 1)
    plt.title(f"Image label={label}")
    plt.imshow(img, cmap="gray")

    plt.subplot(1, 2, 2)
    plt.title("Mask")
    plt.imshow(tumorMask, cmap="gray")

    plt.show()
