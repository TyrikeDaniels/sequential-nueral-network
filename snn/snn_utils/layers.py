#from typing import Optional
import numpy as np

from .settings import *


class BaseLayer:

    """
    Base layer class. Contains basic methods/attributes of layer subclasses
    """

    def __init__(
            self,
            name: str,
            l_type: str,
        ):

        # next_layer
        self.next_layer = None

        # activation matrix
        self.a_matrix = None

        # layer type
        if not isinstance(l_type, str):
            raise TypeError("Type must be a string.")
        self.l_type = l_type

        # Validate name
        if not isinstance(name, str):
            raise TypeError("Name must be a string.")
        self._name = name

    def set_next(self, next_layer):

        # NOTE: next_layer must be a layer type (DenseLayer, DropoutLayer, InputLayer)
        self.next_layer = next_layer

class DenseLayer(BaseLayer):
    def __init__(
            self,
            name : str = "n/a",
            units : int = 1,
            activation : str = LINEAR,
            regularization : tuple[int, float] | None = None
    ):

        # Validate activation function
        if activation not in [RELU, LINEAR, SOFTMAX, SIGMOID]:
            raise ValueError(f"Unsupported activation function: {activation}.")
        self._act = activation

        # Validate regularization parameter
        if regularization is not None:
            try:
                reg_type, weight_decay = regularization
            except Exception:
                raise ValueError(f"Regularization must be ({L1}|{L2}, weight_decay).")
            if reg_type not in (L1, L2):
                raise ValueError(f"Regularization type must be {L1} or {L2}.")
            if weight_decay < 0:
                raise ValueError("Weight decay must be non-negative.")
            self._reg, self._wd = reg_type, float(weight_decay)

        # Validate units
        if not isinstance(units, int) or units <= 0:
            raise ValueError("Units must be an integer >= 0.")
        self._units = units

        # Parent constructor call
        super().__init__(name, "dense")

        # Layer connectivity
        self.prev_layer = None
        self.next_layer = None

        # Weight matrix, bias vector
        self.w_m = None
        self.b_v = None

        # Activation matrix, gradients, error term for backpropagation, and shape
        self.error_term = None
        self.gradient = None

        # Set I/O shapes
        self.input_shape = None
        self.out_shape = (units,)

        # Set flags
        self._shape_flag = False
        self._reg_flag = False

    def _set_gradient(self):

        self.gradient = {
            "weights": np.dot(self.prev_layer.a_matrix.T, self.error_term),
            "biases": np.sum(self.error_term, axis=0, keepdims=True)
        }
        if self._reg == "L2": self.gradient["weights"] += self._wd * self.w_m
        elif self._reg == "L1": self.gradient["weights"] += self._wd * np.sign(self.w_m)

    def _set_oe(self, y : np.ndarray, loss : str):

        if loss == MSE:
            self.error_term = (1 / self.a_matrix.shape[0]) * (self.a_matrix - y)
        elif (( self._act == SOFTMAX and loss == SCCE ) or
              ( self._act == SIGMOID and loss == BCE  )):
            self.error_term = self.a_matrix - y # Cross-Entropy with Softmax: Simplified gradient

    def _set_he(self):

        if hasattr(self.next_layer, "w_m") and self.next_layer.w_m is not None:
            weighted_error = np.dot(self.next_layer.error_term, self.next_layer.w_m.T)
        else:
            # If next layer has no weights (e.g., dropout), just use its error_term directly
            weighted_error = self.next_layer.error_term

        self.error_term = self._activation_derivative() * weighted_error

    def _activation(self, z: np.ndarray):

        if self._act == RELU:
            return np.maximum(0, z)
        elif self._act == LINEAR:
            return z
        elif self._act == SOFTMAX:
            z_exp = np.exp(z - np.max(z, axis=1, keepdims=True))
            return z_exp / np.sum(z_exp, axis=1, keepdims=True)
        elif self._act == SIGMOID:
            return 1 / (1 + np.exp(-z))
        else:
            raise ValueError(f"Unsupported activation function: {self._act}.")

    def _activation_derivative(self):

        if self._act == RELU:
            return (self.a_matrix > 0).astype(float)
        elif self._act == LINEAR:
            return np.ones_like(self.a_matrix)
        elif self._act == SOFTMAX:
            raise ValueError("Softmax derivative is handled implicitly with cross-entropy loss.")
        else:
            raise ValueError(f"Unsupported activation function: {self._act}.")

    def back_prop(self, y : np.ndarray = None, loss : str = None, layer_type : str = "hidden"):

        # Set output layer error
        if layer_type == "output":
            self._set_oe(y, loss) # set output layer error
        elif layer_type == "hidden":
            self._set_he()        # set hidden layer error
        self._set_gradient()

        if not isinstance(self.prev_layer, InputLayer):
            self.prev_layer.back_prop(
                y=y,
                loss=loss,
                layer_type="hidden"
            )

    def forward_prop(self, x : np.ndarray):

        # (m, n) * (n, p) => (m, p) MATRIX MULTIPLICATION YAY
        z = np.dot(x, self.w_m) + self.b_v

        # activation matrix should be of size (m, p)
        self.a_matrix = self._activation(z)

        # pass activation matrix to next layer, return activation matrix otherwise
        if self.next_layer is None:
            return self.a_matrix
        else:
            return self.next_layer.forward_pass(self.a_matrix)

    def reg_loss(self):
        if self._reg is L1: return self._wd * sum(np.abs(self.w_m))
        elif self._reg is L2: return self._wd * sum(self.w_m ** 2)
        return 0.0

    def set_previous(self, prev_layer):

        self.prev_layer = prev_layer
        self.input_shape = prev_layer.out_shape
        self._shape_flag = True

    def set_params(self):

        # Check input shape flag
        if self._shape_flag:

            n = self.input_shape[0] # n.o. neuron activations in previous layer
            p = self._units         # n.o. neurons in current layer

            # He initialization (Draw from normalized pool)
            if self._act == RELU:
                self.w_m = np.random.normal(0, np.sqrt(2 / p), size=(n, p))

            # Xavier initialization (Draw from normalized pool)
            elif self._act == SOFTMAX or self._act == SIGMOID:
                self.w_m = np.random.normal(0, np.sqrt(2 / (n + p)), size=(n, p))

            # Draw from pool [n, p) as small random value
            elif self._act == LINEAR:
                self.w_m = np.random.randn(n, p) * 0.01

            # Set bias vector
            self.b_v = np.zeros((1, p))

        else:
            raise ValueError("input_shape has not been set, call set_previous() method.")

class InputLayer(BaseLayer):

    def __init__(self, name : str, input_shape : tuple):
        super().__init__(name, "input")

        self.a_matrix = None
        self.input_shape = input_shape

        # Set flattening flag
        if len(input_shape) >= 2:
            self._flatten_flag = True
            self.out_shape = (np.prod(input_shape),)
        else:
            self._flatten_flag = False
            self.out_shape = input_shape

    def forward_prop(self, x : np.ndarray):

        # flatten features if flattening flag is set
        if self._flatten_flag:
            x = x.reshape(x.shape[0], -1)
        self.a_matrix = x

        if self.next_layer:
            return self.next_layer.forward_prop(x)
        return x