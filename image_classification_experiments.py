"""
Image Classification Experiments with MLPs and CNNs

This script runs a clean set of Keras/TensorFlow experiments on:
- Fashion-MNIST
- CIFAR-10

It compares:
- Dense MLP baselines
- CNN baselines
- CIFAR-10 MLP neuron-width sensitivity

Outputs:
- results/experiment_summary.csv
- results/neuron_sweep_results.csv
- results/training_curves/*.png

Run:
    python image_classification_experiments.py

Optional examples:
    python image_classification_experiments.py --epochs-fashion 10 --epochs-cifar 15
    python image_classification_experiments.py --skip-cifar
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# -----------------------------------------------------------------------------
# Reproducibility

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# -----------------------------------------------------------------------------
# Paths

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
CURVES_DIR = RESULTS_DIR / "training_curves"

RESULTS_DIR.mkdir(exist_ok=True)
CURVES_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Experiment result schema

@dataclass
class ExperimentResult:
    experiment_name: str
    dataset: str
    architecture: str
    epochs: int
    batch_size: int
    optimizer: str
    learning_rate: float
    train_accuracy: float
    validation_accuracy: float
    test_accuracy: float
    test_loss: float
    parameter_count: int
    notes: str


# -----------------------------------------------------------------------------
# Dataset loading

def load_fashion_mnist() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load Fashion-MNIST and return normalized image tensors."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Add channel dimension for CNN compatibility: (28, 28) -> (28, 28, 1)
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    return x_train, y_train, x_test, y_test


def load_cifar10() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load CIFAR-10 and return normalized image tensors."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # CIFAR-10 labels are shaped (n, 1). Flatten for sparse crossentropy.
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)

    return x_train, y_train, x_test, y_test


# -----------------------------------------------------------------------------
# Model builders

def build_fashion_mnist_mlp() -> keras.Model:
    """Dense MLP baseline for Fashion-MNIST."""
    model = keras.Sequential(
        [
            keras.Input(shape=(28, 28, 1)),
            layers.Flatten(),
            layers.Dense(300, activation="relu"),
            layers.Dense(100, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ],
        name="fashion_mnist_mlp",
    )
    return model


def build_fashion_mnist_cnn() -> keras.Model:
    """Actual CNN baseline for Fashion-MNIST."""
    model = keras.Sequential(
        [
            keras.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.25),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.25),
            layers.Dense(10, activation="softmax"),
        ],
        name="fashion_mnist_cnn",
    )
    return model


def build_cifar10_mlp(hidden_units: int = 512) -> keras.Model:
    """Dense MLP baseline for CIFAR-10."""
    model = keras.Sequential(
        [
            keras.Input(shape=(32, 32, 3)),
            layers.Flatten(),
            layers.Dense(hidden_units, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(hidden_units, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(10, activation="softmax"),
        ],
        name=f"cifar10_mlp_{hidden_units}",
    )
    return model


def build_cifar10_cnn() -> keras.Model:
    """
    Improved CNN baseline for CIFAR-10.

    The earlier rough version used RMSprop with a high learning rate, which
    underperformed. This version uses Adam, batch normalization, dropout, and
    a moderate architecture while staying simple enough for a portfolio repo.
    """
    model = keras.Sequential(
        [
            keras.Input(shape=(32, 32, 3)),

            layers.Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),

            layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.30),

            layers.Conv2D(128, kernel_size=(3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.35),

            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.40),
            layers.Dense(10, activation="softmax"),
        ],
        name="cifar10_cnn",
    )
    return model


def build_cifar10_width_mlp(hidden_units: int) -> keras.Model:
    """MLP used for CIFAR-10 neuron-width sensitivity experiments."""
    model = keras.Sequential(
        [
            keras.Input(shape=(32, 32, 3)),
            layers.Flatten(),
            layers.Dense(hidden_units, activation="relu"),
            layers.Dense(hidden_units, activation="relu"),
            layers.Dense(hidden_units, activation="relu"),
            layers.Dense(hidden_units, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ],
        name=f"cifar10_width_sweep_{hidden_units}",
    )
    return model


# -----------------------------------------------------------------------------
# Training helpers

def compile_model(model: keras.Model, learning_rate: float) -> keras.Model:
    """Compile model using a consistent sparse-label setup."""
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_training_curve(history: keras.callbacks.History, output_path: Path, title: str) -> None:
    """Save training and validation accuracy/loss curves."""
    history_df = pd.DataFrame(history.history)

    fig, ax = plt.subplots(figsize=(9, 5))
    history_df[["accuracy", "val_accuracy"]].plot(ax=ax)
    ax.set_title(f"{title}: Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path.with_name(output_path.stem + "_accuracy.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    history_df[["loss", "val_loss"]].plot(ax=ax)
    ax.set_title(f"{title}: Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path.with_name(output_path.stem + "_loss.png"), dpi=150)
    plt.close(fig)


def train_and_evaluate(
    model: keras.Model,
    dataset_name: str,
    architecture_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    notes: str,
) -> ExperimentResult:
    """Train one model, evaluate it, save curves, and return a result record."""
    model = compile_model(model, learning_rate=learning_rate)

    print("\n" + "=" * 80)
    print(f"Training: {architecture_name} on {dataset_name}")
    print("=" * 80)
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=callbacks,
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

    train_accuracy = float(history.history["accuracy"][-1])
    validation_accuracy = float(history.history["val_accuracy"][-1])

    curve_file = CURVES_DIR / f"{dataset_name.lower()}_{architecture_name.lower().replace(' ', '_')}.png"
    plot_training_curve(history, curve_file, f"{dataset_name} - {architecture_name}")

    return ExperimentResult(
        experiment_name=f"{dataset_name} - {architecture_name}",
        dataset=dataset_name,
        architecture=architecture_name,
        epochs=len(history.history["loss"]),
        batch_size=batch_size,
        optimizer="Adam",
        learning_rate=learning_rate,
        train_accuracy=train_accuracy,
        validation_accuracy=validation_accuracy,
        test_accuracy=float(test_accuracy),
        test_loss=float(test_loss),
        parameter_count=int(model.count_params()),
        notes=notes,
    )


def run_cifar10_neuron_sweep(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    neuron_counts: List[int],
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> pd.DataFrame:
    """Run CIFAR-10 dense-layer width sensitivity experiment."""
    results = []

    for hidden_units in neuron_counts:
        print("\n" + "-" * 80)
        print(f"CIFAR-10 neuron sweep: {hidden_units} hidden units")
        print("-" * 80)

        model = build_cifar10_width_mlp(hidden_units)
        model = compile_model(model, learning_rate=learning_rate)

        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
        )

        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

        results.append(
            {
                "hidden_units": hidden_units,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "train_accuracy": float(history.history["accuracy"][-1]),
                "validation_accuracy": float(history.history["val_accuracy"][-1]),
                "test_accuracy": float(test_accuracy),
                "test_loss": float(test_loss),
                "parameter_count": int(model.count_params()),
            }
        )

    sweep_df = pd.DataFrame(results)
    sweep_df.to_csv(RESULTS_DIR / "neuron_sweep_results.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sweep_df["hidden_units"], sweep_df["test_accuracy"], marker="o")
    ax.set_title("CIFAR-10 MLP Width Sweep")
    ax.set_xlabel("Hidden units per dense layer")
    ax.set_ylabel("Test accuracy")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(CURVES_DIR / "cifar10_neuron_sweep.png", dpi=150)
    plt.close(fig)

    return sweep_df


# -----------------------------------------------------------------------------
# Main experiment runner

def run_experiments(args: argparse.Namespace) -> None:
    """Run selected experiments and save outputs."""
    experiment_results: List[ExperimentResult] = []

    if not args.skip_fashion:
        x_train_fm, y_train_fm, x_test_fm, y_test_fm = load_fashion_mnist()

        experiment_results.append(
            train_and_evaluate(
                model=build_fashion_mnist_mlp(),
                dataset_name="Fashion-MNIST",
                architecture_name="MLP Baseline",
                x_train=x_train_fm,
                y_train=y_train_fm,
                x_test=x_test_fm,
                y_test=y_test_fm,
                epochs=args.epochs_fashion,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                notes="Dense baseline on grayscale clothing images.",
            )
        )

        experiment_results.append(
            train_and_evaluate(
                model=build_fashion_mnist_cnn(),
                dataset_name="Fashion-MNIST",
                architecture_name="CNN Baseline",
                x_train=x_train_fm,
                y_train=y_train_fm,
                x_test=x_test_fm,
                y_test=y_test_fm,
                epochs=args.epochs_fashion,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                notes="Actual convolutional baseline replacing earlier mislabeled dense model.",
            )
        )

    if not args.skip_cifar:
        x_train_cf, y_train_cf, x_test_cf, y_test_cf = load_cifar10()

        experiment_results.append(
            train_and_evaluate(
                model=build_cifar10_mlp(hidden_units=512),
                dataset_name="CIFAR-10",
                architecture_name="MLP Baseline",
                x_train=x_train_cf,
                y_train=y_train_cf,
                x_test=x_test_cf,
                y_test=y_test_cf,
                epochs=args.epochs_cifar,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                notes="Dense baseline that flattens color images and ignores spatial locality.",
            )
        )

        experiment_results.append(
            train_and_evaluate(
                model=build_cifar10_cnn(),
                dataset_name="CIFAR-10",
                architecture_name="CNN Baseline",
                x_train=x_train_cf,
                y_train=y_train_cf,
                x_test=x_test_cf,
                y_test=y_test_cf,
                epochs=args.epochs_cifar,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                notes="Improved convolutional model with batch normalization and dropout.",
            )
        )

        if not args.skip_sweep:
            sweep_df = run_cifar10_neuron_sweep(
                x_train=x_train_cf,
                y_train=y_train_cf,
                x_test=x_test_cf,
                y_test=y_test_cf,
                neuron_counts=args.neuron_counts,
                epochs=args.epochs_sweep,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
            )
            print("\nCIFAR-10 neuron sweep results:")
            print(sweep_df)

    summary_df = pd.DataFrame([asdict(result) for result in experiment_results])
    summary_path = RESULTS_DIR / "experiment_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    metadata = {
        "seed": SEED,
        "tensorflow_version": tf.__version__,
        "keras_version": keras.__version__,
        "summary_file": str(summary_path),
    }

    with open(RESULTS_DIR / "run_metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("\n" + "=" * 80)
    print("Experiment summary")
    print("=" * 80)

    if not summary_df.empty:
        print(summary_df[["experiment_name", "test_accuracy", "test_loss", "parameter_count"]])
        print(f"\nSaved summary to: {summary_path}")
        print(f"Saved training curves to: {CURVES_DIR}")
    else:
        print("No experiments were run.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MLP and CNN image classification experiments on Fashion-MNIST and CIFAR-10."
    )

    parser.add_argument("--epochs-fashion", type=int, default=15)
    parser.add_argument("--epochs-cifar", type=int, default=20)
    parser.add_argument("--epochs-sweep", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)

    parser.add_argument("--skip-fashion", action="store_true")
    parser.add_argument("--skip-cifar", action="store_true")
    parser.add_argument("--skip-sweep", action="store_true")

    parser.add_argument(
        "--neuron-counts",
        nargs="+",
        type=int,
        default=[10, 20, 30, 40, 50, 75, 100],
        help="Hidden-unit sizes for CIFAR-10 MLP width sweep.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    run_experiments(parse_args())
