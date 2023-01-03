import argparse
import csv
import os
import random
import sys
import time
from os.path import join

import numpy as np
import pandas as pd
import tensorflow as tf
from imgaug import augmenters as iaa

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.const import DATA_DIRECTORY, MODELS_DIRECTORY
from config.datasets import get_dataset_params
from config.models import get_model, get_model_params

if not os.path.exists(DATA_DIRECTORY):
    DATA_DIRECTORY = "data"
if not os.path.exists(MODELS_DIRECTORY):
    MODELS_DIRECTORY = "models"


class Train:
    def __init__(self, dataset, architecture, df_path=None):
        self.architecture = architecture
        self.dataset = dataset
        self.set_dataset_params()
        self.set_model_params()
        self.set_model()
        if df_path is None:
            self.uuid = "all_trainset"
            self.fraction = 1
            self.dataframe = None
            self.train_gen = self.datagen.flow_from_directory(
                os.path.join(DATA_DIRECTORY, dataset, "train"),
                target_size=self.target_size[:2],
                color_mode=self.color_mode,
                class_mode="categorical",
                batch_size=self.batch_size,
            )
        else:
            self.uuid = df_path.split("_")[-1].split(".")[0]
            self.fraction = df_path.split("_")[-2]
            self.dataframe = pd.read_csv(df_path)
            self.dataframe["fixed_filename"] = self.dataframe.apply(
                lambda row: join(DATA_DIRECTORY, *row.filename.split("/")[-4:]), axis=1
            )
            self.train_gen = self.datagen.flow_from_dataframe(
                self.dataframe,
                directory="/",
                x_col="fixed_filename",
                y_col="class",
                target_size=self.target_size[:2],
                color_mode=self.color_mode,
                class_mode="categorical",
                batch_size=self.batch_size,
            )

        self.val_gen = self.datagen.flow_from_directory(
            os.path.join(DATA_DIRECTORY, dataset, "val"),
            target_size=self.target_size[:2],
            color_mode=self.color_mode,
            class_mode="categorical",
            batch_size=self.batch_size,
        )
        self.test_gen = self.datagen.flow_from_directory(
            os.path.join(DATA_DIRECTORY, dataset, "test"),
            target_size=self.target_size[:2],
            color_mode=self.color_mode,
            class_mode="categorical",
            batch_size=self.batch_size,
        )

        self.rand_aug = iaa.RandAugment(n=self.augmenter_n, m=self.augmenter_magnitude)
        self.modify_generators()

    def augment_generator(self, generator, training=True):
        while True:
            x, y = next(generator)
            if training:
                yield self.rand_aug(images=x.astype(np.uint8)), y
            else:
                yield x.astype(np.uint8), y

    def modify_generators(self):
        self.train_gen = self.augment_generator(self.train_gen, training=True)
        self.val_gen = self.augment_generator(self.val_gen, training=False)
        self.test_gen = self.augment_generator(self.test_gen, training=False)

    def train_model(self):

        save_filename = f"{self.architecture}_{self.fraction}_{self.uuid}_ckpt"
        save_dir = os.path.join(
            MODELS_DIRECTORY, self.dataset, "teachers", save_filename
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.patience,
                mode="min",
                restore_best_weights=True,
            ),
            tf.keras.callbacks.CSVLogger(
                os.path.join(
                    MODELS_DIRECTORY, self.dataset, "logs", f"{self.uuid}.csv"
                ),
                append=True,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=save_dir,
                save_best_only=False,
                save_weights_only=False,
            ),
        ]

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        self.model.fit(
            x=self.train_gen,
            epochs=self.epochs,
            callbacks=callbacks,
            validation_data=self.val_gen,
            steps_per_epoch=int(self.train_size / self.batch_size),
            validation_steps=int(self.val_size / self.batch_size),
            use_multiprocessing=False,
            workers=1,
            max_queue_size=10,
        )

    def evaluate_and_save(self, filename="teachers.csv", loss_name="loss"):

        test_result = self.model.evaluate(
            x=self.test_gen,
            return_dict=True,
            steps=int(self.test_size / self.batch_size),
        )
        val_result = self.model.evaluate(
            x=self.val_gen, return_dict=True, steps=int(self.val_size / self.batch_size)
        )
        train_result = self.model.evaluate(
            x=self.train_gen,
            return_dict=True,
            steps=int(self.train_size / self.batch_size),
        )

        test_accuracy = test_result.get("accuracy")
        test_loss = test_result.get(loss_name)
        train_accuracy = train_result.get("accuracy")
        train_loss = train_result.get(loss_name)
        val_accuracy = val_result.get("accuracy")
        val_loss = val_result.get(loss_name)

        save_filename = f"{self.architecture}_{self.fraction}_{self.uuid}"
        save_dir = os.path.join(
            MODELS_DIRECTORY, self.dataset, "teachers", save_filename
        )
        self.model.save(save_dir)

        csv_log_filename = os.path.join(MODELS_DIRECTORY, filename)
        fileEmpty = not os.path.isfile(csv_log_filename)
        with open(csv_log_filename, "a") as csvfile:
            headers = [
                "model_location",
                "dataset",
                "architecture",
                "fraction",
                "uuid",
                "train_loss",
                "train_accuracy",
                "val_loss",
                "val_accuracy",
                "test_loss",
                "test_accuracy",
            ]
            writer = csv.DictWriter(
                csvfile, delimiter=",", lineterminator="\n", fieldnames=headers
            )
            if fileEmpty:
                writer.writeheader()  # file doesn't exist yet, write a header

            writer.writerow(
                {
                    "model_location": save_dir,
                    "dataset": self.dataset,
                    "architecture": self.architecture,
                    "fraction": str(self.fraction),
                    "uuid": self.uuid,
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "test_loss": test_loss,
                    "test_accuracy": test_accuracy,
                }
            )

    def set_model_params(self):
        model_params = get_model_params(self.architecture)
        self.epochs = model_params["epochs"]
        self.lr = model_params["lr"]
        self.batch_size = model_params["batch_size"]
        self.patience = model_params["patience"]
        self.augmenter_n = model_params["augmenter_n"]
        self.augmenter_magnitude = model_params["augmenter_magnitude"]

    def set_dataset_params(self):
        dataset_params = get_dataset_params(self.dataset)
        self.target_size = dataset_params["target_size"]
        self.color_mode = dataset_params["color_mode"]
        self.classes = dataset_params["classes"]
        self.datagen = dataset_params["datagen"]
        self.train_size = dataset_params["train_size"]
        self.val_size = dataset_params["val_size"]
        self.test_size = dataset_params["test_size"]

    def set_model(self):
        self.model = get_model(self.architecture, self.target_size, self.classes)


def check_model_exists(dataset, architecture, directory):
    uuid = directory.split("_")[-1].split(".")[0]
    fraction = directory.split("_")[-2]
    teachers = os.path.join(MODELS_DIRECTORY, "teachers.csv")
    if os.path.exists(teachers):
        teachers = pd.read_csv(teachers)
        exists_teachers = (
            (teachers["dataset"] == dataset)
            & (teachers["architecture"] == architecture)
            & (teachers["uuid"] == uuid)
        ).any()
    else:
        exists_teachers = False

    training = os.path.join(MODELS_DIRECTORY, "training.csv")
    if os.path.exists(os.path.join(MODELS_DIRECTORY, "training.csv")):
        training = pd.read_csv(training)
        exists_training = (
            (training["dataset"] == dataset)
            & (training["architecture"] == architecture)
            & (training["uuid"] == uuid)
        ).any()
        if exists_training:
            checkpoint_filename = f"{architecture}_{fraction}_{uuid}_ckpt"
            checkpoint_dir = os.path.join(
                MODELS_DIRECTORY, dataset, checkpoint_filename
            )
            exists_training = os.path.isdir(checkpoint_dir)
            if exists_training:
                exists_training = os.path.getmtime(checkpoint_dir) + 1 >= time.time()

    else:
        exists_training = False

    if exists_teachers:
        print("model already trained!")
    elif exists_training:
        print("model during training!")
    return exists_teachers or exists_training


def add_to_training(dataset, architecture, directory):
    csv_log_filename = os.path.join(MODELS_DIRECTORY, "training.csv")
    fileEmpty = not os.path.isfile(csv_log_filename)
    uuid = directory.split("_")[-1].split(".")[0]
    with open(csv_log_filename, "a") as csvfile:
        headers = ["dataset", "architecture", "uuid"]
        writer = csv.DictWriter(
            csvfile, delimiter=",", lineterminator="\n", fieldnames=headers
        )
        if fileEmpty:
            writer.writeheader()  # file doesn't exist yet, write a header

        writer.writerow(
            {"dataset": dataset, "architecture": architecture, "uuid": uuid}
        )


def train_all(dataset, architecture):
    train(dataset, architecture)


def train(dataset, architecture, fraction=None):
    df_files = os.listdir(os.path.join(MODELS_DIRECTORY, dataset, "subsets"))
    print(df_files)
    if fraction:
        df_files = [f for f in df_files if float(f.split("_")[-2]) / 10 == fraction]
    print(df_files)
    random.shuffle(df_files)
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        for df_file in df_files:
            df_path = os.path.join(MODELS_DIRECTORY, dataset, "subsets", df_file)
            if not check_model_exists(dataset, architecture, df_path):
                add_to_training(dataset, architecture, df_path)
                train = Train(
                    dataset=dataset, df_path=df_path, architecture=architecture
                )
                train.train_model()
                train.evaluate_and_save()
                tf.keras.backend.clear_session()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="dataset name")
    parser.add_argument("architecture", help="architecture name")
    parser.add_argument(
        "--all", help="train on all possible subsets", action="store_true"
    )
    parser.add_argument("--fraction", default=None)
    args = parser.parse_args()

    print(args)
    if args.all:
        train_all(dataset=args.dataset, architecture=args.architecture)
    else:
        train(
            dataset=args.dataset,
            architecture=args.architecture,
            fraction=float(args.fraction),
        )
