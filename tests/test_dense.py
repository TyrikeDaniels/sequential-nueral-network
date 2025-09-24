import pytest
import numpy as np

from snn.snn_utils.layers import DenseLayer
from snn.snn_utils.settings import RELU, LINEAR, SOFTMAX, SIGMOID, L1, L2



# ---------- VALID INITIALIZATION TESTS ----------
def test_default_initialization():
    layer = DenseLayer()
    assert layer._act == LINEAR
    assert layer._units == 1
    assert layer.out_shape == (1,)
    assert layer._shape_flag is False
    assert layer._reg_flag is False


@pytest.mark.parametrize("activation", [RELU, LINEAR, SOFTMAX, SIGMOID])
def test_valid_activations(activation):
    layer = DenseLayer(units=5, activation=activation)
    assert layer._act == activation
    assert layer._units == 5
    assert layer.out_shape == (5,)


@pytest.mark.parametrize("reg", [(L1, 0.01), (L2, 0.1)])
def test_valid_regularization(reg):
    reg_type, wd = reg
    layer = DenseLayer(units=3, activation=LINEAR, regularization=reg)
    assert layer._reg == reg_type
    assert layer._wd == wd


# ---------- INVALID INITIALIZATION TESTS ----------
def test_invalid_activation_raises():
    with pytest.raises(ValueError, match="Unsupported activation function"):
        DenseLayer(activation="not_an_act")


def test_invalid_regularization_type_raises():
    with pytest.raises(ValueError, match="Regularization type must be"):
        DenseLayer(regularization=("bad_type", 0.01))


def test_invalid_regularization_format_raises():
    with pytest.raises(ValueError, match="Regularization must be"):
        DenseLayer(regularization="not_a_tuple")


def test_negative_weight_decay_raises():
    with pytest.raises(ValueError, match="Weight decay must be non-negative"):
        DenseLayer(regularization=(L1, -0.5))


@pytest.mark.parametrize("bad_units", [0, -3, 2.5, "ten", None])
def test_invalid_units_raises(bad_units):
    with pytest.raises(ValueError, match="Units must be an integer"):
        DenseLayer(units=bad_units)
