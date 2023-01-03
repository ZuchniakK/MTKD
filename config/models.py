import tensorflow as tf


def get_model_params(architecture):
    if architecture == "test":
        epochs = 1
        lr = 0.005
        batch_size = 128
        patience = 20
        augmenter_n = 3
        augmenter_magnitude = 7
    elif architecture == "resnet50_v2":
        epochs = 300
        lr = 0.001
        batch_size = 128
        patience = 30
        augmenter_n = 3
        augmenter_magnitude = 7
    elif architecture == "resnet50_v2_aug3":
        epochs = 300
        lr = 0.001
        batch_size = 128
        patience = 30
        augmenter_n = 3
        augmenter_magnitude = 3
    elif architecture == "resnet50_v2_aug0":
        epochs = 300
        lr = 0.001
        batch_size = 128
        patience = 30
        augmenter_n = 3
        augmenter_magnitude = 0
    elif architecture == "efficientnetb0":
        epochs = 300
        lr = 0.001
        batch_size = 64
        patience = 30
        augmenter_n = 3
        augmenter_magnitude = 7
    elif architecture == "efficientnetb3":
        epochs = 300
        lr = 0.001
        batch_size = 64
        patience = 30
        augmenter_n = 3
        augmenter_magnitude = 7
    elif architecture == "densenet121":
        epochs = 300
        lr = 0.005
        batch_size = 128
        patience = 30
        augmenter_n = 3
        augmenter_magnitude = 7
    elif architecture == "mobilenetv3large":
        epochs = 200
        lr = 0.005
        batch_size = 32
        patience = 20
        augmenter_n = 3
        augmenter_magnitude = 7

    else:
        raise NotImplementedError(
            f'model architecture "{architecture}" not implemented'
        )

    params = {
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "patience": patience,
        "augmenter_n": augmenter_n,
        "augmenter_magnitude": augmenter_magnitude,
    }

    return params


def get_model(architecture, target_size, classes):
    if architecture == "test":
        model = tf.keras.Sequential()
        model.add(
            tf.keras.layers.Conv2D(
                filters=16,
                kernel_size=3,
                padding="same",
                activation="relu",
                input_shape=target_size,
            )
        )
        model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(classes, activation="softmax"))

    elif architecture in ["resnet50_v2", "resnet50_v2_aug3"]:
        resnet50_v2 = tf.keras.applications.ResNet50V2(
            weights=None,
            include_top=True,
            pooling="avg",
            input_shape=target_size,
            classes=classes,
        )
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(target_size),
                tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1),
                resnet50_v2,
            ]
        )

    elif architecture == "densenet121":
        densenet121 = tf.keras.applications.densenet.DenseNet121(
            weights=None,
            include_top=True,
            pooling="avg",
            input_shape=target_size,
            classes=classes,
        )
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(target_size),
                tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1),
                densenet121,
            ]
        )

    elif architecture == "efficientnetb0":
        efficientnetb0 = tf.keras.applications.efficientnet.EfficientNetB0(
            weights=None,
            include_top=True,
            pooling="avg",
            input_shape=target_size,
            classes=classes,
        )
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(target_size),
                tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1),
                efficientnetb0,
            ]
        )

    elif architecture == "efficientnetb3":
        efficientnetb3 = tf.keras.applications.efficientnet.EfficientNetB3(
            weights=None,
            include_top=True,
            pooling="avg",
            input_shape=target_size,
            classes=classes,
        )
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(target_size),
                tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1),
                efficientnetb3,
            ]
        )

    elif architecture == "mobilenetv3large":
        mobilenetv3large = tf.keras.applications.MobileNetV3Large(
            weights=None,
            include_top=True,
            pooling="avg",
            input_shape=target_size,
            classes=classes,
        )
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(target_size),
                tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1),
                mobilenetv3large,
            ]
        )
    else:
        raise NotImplementedError(
            f'model architecture "{architecture}" not implemented'
        )

    return model
