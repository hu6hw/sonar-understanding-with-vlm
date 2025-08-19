import h5py
import matplotlib.pyplot as plt
# import plotext as plt
import pandas as pd
import numpy as np
from ollama import chat
from PIL import Image
import io
import base64

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

    # df = pd.DataFrame(image)

    # # Plot using plotext
    # plt.clear_figure()
    # plt.heatmap(df)
    # plt.show()

    plt.imshow(image, cmap="gray")
    # plt.title("Image 0")
    plt.axis("off")

    image_stream = io.BytesIO()
    plt.savefig(image_stream, format="png", bbox_inches="tight")
    image_stream.seek(0)

    # Convert the plot to base64
    base64_image = base64.b64encode(image_stream.read()).decode("utf-8")

    print(base64_image)

with h5py.File(HDF5FILE, "r") as f:
    f.visititems(explore)

    x_train = f["x_train"]
    y_train = f["y_train"]
    class_names = f["class_names"]
    assert isinstance(x_train, h5py.Dataset)
    assert isinstance(y_train, h5py.Dataset)
    assert isinstance(class_names, h5py.Dataset)

    selected_class = y_train[0]
    print(selected_class)
    print(class_names[...])
    
    image = x_train[0].squeeze()
    print_np_array_img(image)

