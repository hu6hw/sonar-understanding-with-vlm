import h5py
import matplotlib.pyplot as plt
import numpy as np

HDF5FILE = "marine-debris-watertank-classification-96x96.hdf5"

def explore(name, obj):
    print(f"Name: {name}")
    if isinstance(obj, h5py.Dataset):
        print(f" - Dataset, shape: {obj.shape}, dtype: {obj.dtype}")
        # print(obj[0])
    elif isinstance(obj, h5py.Group):
        print(" - Group")
    for key, val in obj.attrs.items():
        print(f"   - Attribute: {key} => {val}")

def print_np_array_img(image: np.ndarray):
    """Take a numpy nd array and print using matplotlib

    Args:
        image (numpy.ndarray)
    """
    plt.imshow(image, cmap="gray")
    plt.title("Image 0")
    plt.axis("off")
    plt.show()

with h5py.File(HDF5FILE, "r") as f:
    f.visititems(explore)

    x_train = f["x_train"]
    assert isinstance(x_train, h5py.Dataset)
    
    image = x_train[0].squeeze()
    print_np_array_img(image)
