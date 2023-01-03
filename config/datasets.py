import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def get_dataset_params(dataset):
    if dataset == "mnist":
        target_size = (28, 28, 1)
        color_mode = "grayscale"
        classes = 10
        datagen = ImageDataGenerator(
            rotation_range=5,
            width_shift_range=0.02,
            height_shift_range=0.02,
            rescale=1.0 / 255,
            shear_range=0.02,
            zoom_range=0.02,
            horizontal_flip=False,
            fill_mode="nearest",
        )
        train_size = 40097
        val_size = 9903
        test_size = 10000

    elif dataset == "fmnist":
        target_size = (28, 28, 1)
        color_mode = "grayscale"
        classes = 10
        datagen = ImageDataGenerator(
            rotation_range=5,
            width_shift_range=0.02,
            height_shift_range=0.02,
            rescale=1.0 / 255,
            shear_range=0.02,
            zoom_range=0.02,
            horizontal_flip=True,
            fill_mode="nearest",
        )
        train_size = 40097
        val_size = 9903
        test_size = 10000

    elif dataset == "emnist":
        target_size = (28, 28, 1)
        color_mode = "grayscale"
        classes = 47
        datagen = ImageDataGenerator(
            rotation_range=5,
            width_shift_range=0.02,
            height_shift_range=0.02,
            rescale=1.0 / 255,
            shear_range=0.02,
            zoom_range=0.02,
            horizontal_flip=False,
            fill_mode="nearest",
        )
        train_size = 40097
        val_size = 9903
        test_size = 10000

    elif dataset == "cifar10":
        target_size = (32, 32, 3)
        color_mode = "rgb"
        classes = 10
        datagen = ImageDataGenerator(dtype=np.uint8)
        train_size = 40097
        val_size = 9903
        test_size = 10000

    elif dataset == "cifar100":
        target_size = (32, 32, 3)
        color_mode = "rgb"
        classes = 100
        datagen = ImageDataGenerator(dtype=np.uint8)
        train_size = 40013
        val_size = 9987
        test_size = 10000

    elif dataset == "corrosion_320":
        target_size = (240, 320, 3)
        color_mode = "rgb"
        classes = 2
        datagen = ImageDataGenerator(dtype=np.uint8)
        train_size = 10181
        val_size = 1342
        test_size = 986

    else:
        raise NotImplementedError(f'dataset "{dataset}" not implemented')

    params = {
        "target_size": target_size,
        "color_mode": color_mode,
        "classes": classes,
        "datagen": datagen,
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
    }

    return params
