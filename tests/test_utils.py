# tests/test_utils.py

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pytest
import tensorflow as tf
from tensorflow import keras

from src.utils import (
    CLASS_NAMES,
    Cifar10Data,
    build_cifar10_cnn,
    classification_report_str,
    compile_model,
    confusion_matrix_array,
    create_data_augmentation,
    evaluate_model,
    load_cifar10,
    predict_classes,
    set_global_seed,
)


@pytest.fixture(scope="session")
def small_dummy_data() -> Dict[str, np.ndarray]:
    """
    Create a small synthetic dataset for fast tests.

    The shapes are compatible with CIFAR-10:
    - images: (N, 32, 32, 3)
    - labels: (N,)
    """
    set_global_seed(123)
    n_samples: int = 64
    images = np.random.randint(
        low=0,
        high=256,
        size=(n_samples, 32, 32, 3),
        dtype=np.uint8,
    )
    labels = np.random.randint(
        low=0,
        high=len(CLASS_NAMES),
        size=(n_samples,),
        dtype=np.int32,
    )
    return {"x": images.astype("float32"), "y": labels}


def test_set_global_seed_reproducible() -> None:
    """
    Setting the same global seed should result in identical random tensors.
    """
    set_global_seed(42)
    a = tf.random.uniform(shape=(3, 3))

    set_global_seed(42)
    b = tf.random.uniform(shape=(3, 3))

    assert np.allclose(a.numpy(), b.numpy())


def test_load_cifar10_shapes_and_range() -> None:
    """
    CIFAR-10 loader should return correct shapes and types.

    Also verifies that normalization works when enabled.
    """
    data: Cifar10Data = load_cifar10(normalize=False)

    assert data.x_train.shape[1:] == (32, 32, 3)
    assert data.x_test.shape[1:] == (32, 32, 3)
    assert data.x_train.dtype == np.float32
    assert data.x_test.dtype == np.float32
    assert data.y_train.ndim == 1
    assert data.y_test.ndim == 1
    assert len(data.y_train) == data.x_train.shape[0]
    assert len(data.y_test) == data.x_test.shape[0]

    # Check normalize=True produces values within [0, 1]
    data_norm: Cifar10Data = load_cifar10(normalize=True)
    assert data_norm.x_train.min() >= 0.0
    assert data_norm.x_train.max() <= 1.0 + 1e-5


def test_create_data_augmentation_type_and_shape(small_dummy_data: Dict[str, np.ndarray]) -> None:
    """
    Data augmentation should be a Keras model and preserve image shape.
    """
    aug = create_data_augmentation()
    assert isinstance(aug, keras.Model)

    x = small_dummy_data["x"]
    augmented = aug(x, training=True)

    assert augmented.shape == x.shape


def test_build_cifar10_cnn_shapes_and_classes() -> None:
    """
    The CIFAR-10 CNN model should accept the correct input shape and
    output logits with the expected number of classes.
    """
    model = build_cifar10_cnn()
    assert model.input_shape[1:] == (32, 32, 3)
    assert model.output_shape[-1] == len(CLASS_NAMES)
    assert model.count_params() > 0


def test_compile_model_sets_optimizer_and_loss() -> None:
    """
    compile_model should attach an optimizer, loss and at least one metric.

    In Keras 3, `model.metrics` typically contains:
    - a loss tracker ('loss')
    - a compiled metrics container ('compile_metrics')

    The actual metric objects (e.g. accuracy) live inside
    `model.compiled_metrics.metrics`.
    """
    model = build_cifar10_cnn()
    compile_model(model, learning_rate=1e-3)

    # Optimizer and loss should be set
    assert isinstance(model.optimizer, keras.optimizers.Optimizer)
    assert model.loss is not None

    # There should be a compiled_metrics container with at least one metric
    compiled_metrics = getattr(model, "compiled_metrics", None)
    assert compiled_metrics is not None
    assert len(compiled_metrics.metrics) > 0

    # At least one metric should contain "accuracy" in its name
    metric_names = [m.name for m in compiled_metrics.metrics]
    assert any("accuracy" in name.lower() for name in metric_names)
    assert isinstance(model.optimizer, keras.optimizers.Optimizer)
    assert model.loss is not None
    metric_names = [m.name for m in model.metrics]
    assert "accuracy" in metric_names


def test_train_and_evaluate_smoke(small_dummy_data: Dict[str, np.ndarray]) -> None:
    """
    Quick smoke test: training for 1 epoch on synthetic data
    should run without errors and evaluation should return floats.
    """
    x = small_dummy_data["x"]
    y = small_dummy_data["y"]

    model = build_cifar10_cnn()
    compile_model(model, learning_rate=1e-3)

    history = model.fit(
        x,
        y,
        batch_size=16,
        epochs=1,
        validation_split=0.2,
        verbose=0,
    )
    assert len(history.history["loss"]) == 1

    metrics = evaluate_model(model, x, y)
    assert "loss" in metrics and "accuracy" in metrics
    assert isinstance(metrics["loss"], float)
    assert isinstance(metrics["accuracy"], float)


def test_predict_classes_shape_and_range(small_dummy_data: Dict[str, np.ndarray]) -> None:
    """
    predict_classes should return one class index per input sample,
    and all indices must be within the valid class range.
    """
    x = small_dummy_data["x"]
    y = small_dummy_data["y"]

    model = build_cifar10_cnn()
    compile_model(model, learning_rate=1e-3)

    # Train very briefly so predictions are not completely random
    model.fit(x, y, batch_size=16, epochs=1, validation_split=0.2, verbose=0)

    preds = predict_classes(model, x, batch_size=32)
    assert preds.shape == y.shape
    assert preds.dtype == np.int64 or preds.dtype == np.int32
    assert preds.min() >= 0
    assert preds.max() < len(CLASS_NAMES)


def test_classification_report_contains_class_names() -> None:
    """
    The classification report string should contain at least a few class names.
    """
    y_true = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y_pred = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    report = classification_report_str(y_true, y_pred, target_names=CLASS_NAMES)
    assert isinstance(report, str)
    # spot check for some class names in the text
    assert "airplane" in report
    assert "truck" in report


def test_compile_model_sets_optimizer_and_loss() -> None:
    """
    compile_model should attach an optimizer, a loss and at least one metric.

    We keep this test robust across different Keras versions by checking:
    - optimizer is set
    - loss is set
    - model.metrics_names has at least 2 entries (loss + one metric)
    """
    model = build_cifar10_cnn()
    compile_model(model, learning_rate=1e-3)

    # Optimizer and loss should be set
    assert isinstance(model.optimizer, keras.optimizers.Optimizer)
    assert model.loss is not None

    # metrics_names contains the names returned by model.evaluate(...)
    metrics_names = list(model.metrics_names)  # e.g. ['loss', 'accuracy'] or ['loss', 'metric_1']
    assert len(metrics_names) >= 2  # at least loss + one metric
    # First metric is usually 'loss'
    assert metrics_names[0] == "loss"