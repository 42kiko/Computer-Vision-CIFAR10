import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import tensorflow as tf
from tensorflow import keras

import src.utils as utils


# ---------------------------------------------------------------------------
# Reproducibility and basic constants
# ---------------------------------------------------------------------------

def test_set_global_seed_reproducible() -> None:
    """
    set_global_seed should make NumPy and TensorFlow randomness reproducible.
    """
    utils.set_global_seed(123)
    a1 = np.random.rand(5)
    t1 = tf.random.uniform((3,))

    utils.set_global_seed(123)
    a2 = np.random.rand(5)
    t2 = tf.random.uniform((3,))

    assert np.allclose(a1, a2)
    assert np.allclose(t1.numpy(), t2.numpy())


def test_class_names_and_num_classes_consistent() -> None:
    """
    CLASS_NAMES and NUM_CLASSES should be consistent with each other.
    """
    assert len(utils.CLASS_NAMES) == utils.NUM_CLASSES
    assert len(utils.CLASS_NAMES_EMOJI) == utils.NUM_CLASSES


# ---------------------------------------------------------------------------
# Image enhancement
# ---------------------------------------------------------------------------

def test_upscale_and_super_sharpen_shape_and_dtype() -> None:
    """
    upscale_and_super_sharpen should upscale and return a uint8 image.
    """
    img = (np.random.rand(32, 32, 3) * 255).astype("uint8")

    out = utils.upscale_and_super_sharpen(img, scale=4)

    assert out.dtype == np.uint8
    assert out.shape[0] == img.shape[0] * 4
    assert out.shape[1] == img.shape[1] * 4
    assert out.shape[2] == 3
    assert out.min() >= 0
    assert out.max() <= 255


# ---------------------------------------------------------------------------
# Plotly figure export
# ---------------------------------------------------------------------------

class DummyFig:
    """
    Minimal stand-in for a Plotly Figure used to test save_fig
    without requiring kaleido or Plotly itself.
    """

    def __init__(self) -> None:
        self.html_written_to: Path | None = None
        self.png_written_to: Path | None = None

    def write_html(self, path: str, include_plotlyjs: str = "cdn") -> None:  # noqa: ARG002
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("<html></html>", encoding="utf-8")
        self.html_written_to = p

    def write_image(self, path: str, scale: int = 1) -> None:  # noqa: ARG002
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"PNG")
        self.png_written_to = p


def test_save_fig_creates_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    save_fig should create HTML and PNG files in the configured directories.
    """
    # Redirect PLOTS_DIR and DOCS_DIR to a temporary directory
    monkeypatch.setattr(utils, "PLOTS_DIR", tmp_path / "plots")
    monkeypatch.setattr(utils, "DOCS_DIR", tmp_path / "docs")

    fig = DummyFig()
    utils.save_fig(fig, name="test_plot", scale=2)

    html_path = tmp_path / "docs" / "test_plot.html"
    png_path = tmp_path / "plots" / "test_plot.png"

    assert html_path.exists()
    assert png_path.exists()


# ---------------------------------------------------------------------------
# CIFAR-10 loading
# ---------------------------------------------------------------------------

def test_load_cifar10_shapes_and_range() -> None:
    """
    load_cifar10 should return splits of expected shape and range.

    If CIFAR-10 cannot be loaded (e.g. no internet to download it),
    the test is skipped instead of failing.
    """
    try:
        data = utils.load_cifar10(normalize=True)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"CIFAR-10 not available: {exc}")

    assert data.x_train.shape[1:] == utils.INPUT_SHAPE
    assert data.x_test.shape[1:] == utils.INPUT_SHAPE
    assert data.y_train.ndim == 1
    assert data.y_test.ndim == 1

    assert data.x_train.min() >= 0.0
    assert data.x_train.max() <= 1.0
    assert data.x_test.min() >= 0.0
    assert data.x_test.max() <= 1.0


# ---------------------------------------------------------------------------
# Data augmentation
# ---------------------------------------------------------------------------

def test_create_data_augmentation_preserves_shape() -> None:
    """
    create_data_augmentation should return a model that preserves the input shape.
    """
    aug = utils.create_data_augmentation()
    x = tf.zeros((4,) + utils.INPUT_SHAPE)
    y = aug(x, training=True)

    assert y.shape == x.shape


# ---------------------------------------------------------------------------
# Model architecture and compilation
# ---------------------------------------------------------------------------

def test_build_cifar10_cnn_shapes() -> None:
    """
    build_cifar10_cnn should create a model with correct input and output shapes.
    """
    model = utils.build_cifar10_cnn()
    assert model.input_shape[1:] == utils.INPUT_SHAPE
    assert model.output_shape[1:] == (utils.NUM_CLASSES,)


def test_compile_model_sets_optimizer_and_accuracy_metric() -> None:
    """
    compile_model should attach an optimizer, loss and at least one metric.

    We keep this robust across different Keras versions by:
    - checking that optimizer and loss are set
    - running a tiny evaluate() and asserting it returns >= 2 values
      (loss + at least one metric)
    """
    model = utils.build_cifar10_cnn()
    utils.compile_model(model, learning_rate=1e-3)

    # Optimizer and loss should be set
    assert isinstance(model.optimizer, keras.optimizers.Optimizer)
    assert model.loss is not None

    # Run a tiny evaluation to see how many outputs we get
    x = np.random.rand(4, *utils.INPUT_SHAPE).astype("float32")
    y = np.random.randint(0, utils.NUM_CLASSES, size=(4,), dtype="int32")

    results = model.evaluate(x, y, batch_size=2, verbose=0)

    # Keras may return a scalar or a list/tuple depending on config/version
    if not isinstance(results, (list, tuple)):
        results = [results]

    # We expect at least loss + one additional metric
    assert len(results) >= 2


def test_create_default_callbacks_types() -> None:
    """
    create_default_callbacks should return ReduceLROnPlateau and EarlyStopping.
    """
    callbacks = utils.create_default_callbacks()
    assert len(callbacks) == 2
    assert any(isinstance(cb, keras.callbacks.ReduceLROnPlateau) for cb in callbacks)
    assert any(isinstance(cb, keras.callbacks.EarlyStopping) for cb in callbacks)


# ---------------------------------------------------------------------------
# Training and evaluation (lightweight)
# ---------------------------------------------------------------------------

def _build_tiny_model() -> keras.Model:
    """
    Build a very small model for fast tests.
    """
    model = utils.build_cifar10_cnn()
    utils.compile_model(model, learning_rate=1e-3)
    return model


def test_train_model_runs_with_validation_split() -> None:
    """
    train_model should run end-to-end with a small random dataset.
    """
    model = _build_tiny_model()
    utils.set_global_seed(123)

    x_train = np.random.rand(64, *utils.INPUT_SHAPE).astype("float32")
    y_train = np.random.randint(0, utils.NUM_CLASSES, size=(64,), dtype="int32")

    history = utils.train_model(
        model,
        x_train=x_train,
        y_train=y_train,
        batch_size=16,
        epochs=1,
        validation_split=0.2,
    )

    assert isinstance(history, keras.callbacks.History)
    assert "loss" in history.history


def test_evaluate_model_returns_metrics_dict() -> None:
    """
    evaluate_model should return a dict with 'loss' and 'accuracy' keys.
    """
    model = _build_tiny_model()
    x = np.random.rand(16, *utils.INPUT_SHAPE).astype("float32")
    y = np.random.randint(0, utils.NUM_CLASSES, size=(16,), dtype="int32")

    metrics = utils.evaluate_model(model, x_test=x, y_test=y)
    assert set(metrics.keys()) == {"loss", "accuracy"}
    assert isinstance(metrics["loss"], float)
    assert isinstance(metrics["accuracy"], float)


# ---------------------------------------------------------------------------
# Model + history persistence
# ---------------------------------------------------------------------------

def test_save_and_load_model_and_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    save_model_with_history + load_model + load_history should persist and restore
    files correctly (paths are redirected to a temporary directory).
    """
    # Redirect directories to tmp_path
    monkeypatch.setattr(utils, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(utils, "RESULTS_DIR", tmp_path / "results")

    model = _build_tiny_model()

    # Fake a minimal history object
    history = keras.callbacks.History()
    history.history = {"loss": [1.0, 0.9], "accuracy": [0.5, 0.6]}

    utils.save_model_with_history(model, history, name="test_model")

    # Files should exist
    model_path = tmp_path / "models" / "test_model.keras"
    history_path = tmp_path / "results" / "history_test_model.json"
    assert model_path.exists()
    assert history_path.exists()

    # Loaded model should be a valid Keras model
    loaded_model = utils.load_model("test_model")
    assert isinstance(loaded_model, keras.Model)

    # Loaded history should match the saved one
    loaded_history = utils.load_history("test_model")
    assert loaded_history.history == history.history


# ---------------------------------------------------------------------------
# Prediction and reporting helpers
# ---------------------------------------------------------------------------

def test_predict_classes_shape_and_range() -> None:
    """
    predict_classes should return a 1D array of class indices in the valid range.
    """
    model = _build_tiny_model()
    x = np.random.rand(10, *utils.INPUT_SHAPE).astype("float32")

    preds = utils.predict_classes(model, x=x, batch_size=5)

    assert preds.shape == (10,)
    assert preds.min() >= 0
    assert preds.max() < utils.NUM_CLASSES


def test_classification_report_str_non_empty() -> None:
    """
    classification_report_str should return a non-empty text report.
    """
    y_true = np.array([0, 1, 2, 2, 1])
    y_pred = np.array([0, 2, 2, 1, 1])

    report = utils.classification_report_str(y_true, y_pred, target_names=["a", "b", "c"])
    assert isinstance(report, str)
    assert "precision" in report
    assert "recall" in report


def test_confusion_matrix_array_shape_and_normalization() -> None:
    """
    confusion_matrix_array should return an array of shape (NUM_CLASSES, NUM_CLASSES),
    optionally row-normalised.
    """
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])

    cm = utils.confusion_matrix_array(y_true, y_pred, normalize=False)
    assert cm.shape == (utils.NUM_CLASSES, utils.NUM_CLASSES)

    cm_norm = utils.confusion_matrix_array(y_true, y_pred, normalize=True)
    assert cm_norm.shape == (utils.NUM_CLASSES, utils.NUM_CLASSES)

    # Rows with any samples should sum (approximately) to 1.0
    row_sums = cm_norm.sum(axis=1)
    non_zero_rows = row_sums > 0
    assert np.allclose(row_sums[non_zero_rows], 1.0, atol=1e-6)