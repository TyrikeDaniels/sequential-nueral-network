from tqdm import tqdm

from layers import *
from optimizers import *
from settings import *

class SequentialNN:

    def __init__(self):

        self._optimizer = None
        self._loss = None
        self._y = None
        self._x = None

        self._compile_flag = None
        self._layers = []
        self._epoch_losses = []

    def _add_layer(self, layer):
        self._layers.append(layer)

    def _connect_layers(self):

        # Iterate though layers
        num_layers = len(self._layers)
        for i in range(1, num_layers):

            # Identify head/tail layer
            tail_layer = self._layers[i - 1]
            head_layer = self._layers[i]

            # Set next and previous layers of head and tail (respectively)
            tail_layer.set_next(head_layer)
            head_layer.set_previous(tail_layer)
            head_layer.set_params()

    def compile(
        self,
        training_data,
        optimizer,
        loss=MSE,
    ):

        # Optimizer invariant
        # valid_opt = (SGD, SGDWithMomentum, RMSProp, AdaptiveMovementEstimation)
        # if not isinstance(optimizer, valid_opt):
        #     raise ValueError(f"Required optimizer argument missing. Choose from {valid_opt}.")
        # self._optimizer = optimizer

        # loss function invariant
        if loss not in [MSE, SCCE, BCE]:
            raise ValueError(f"Unsupported loss function: {loss}")
        self._loss = loss

        # set training data and invariants
        x_train, y_train = training_data
        if not isinstance(x_train, np.ndarray) or not isinstance(y_train, np.ndarray):
            raise ValueError("Ensure training data is of type np.ndarray.")
        if x_train.ndim < 2:
            raise ValueError(f"x_train must be at least 2-dimensional, got shape {x_train.shape}")
        if y_train.ndim != 2:
            raise ValueError(f"y_train must be exactly 2-dimensional, got shape {y_train.shape}")
        self._x, self._y = x_train, y_train

        # set optimizer and invariants
        if isinstance(optimizer, (SGD, SGDWithMomentum, RMSProp, AdaptiveMovementEstimation)):
            self._optimizer = optimizer
        else:
            raise ValueError(f"Invalid optimizer passed.")

        # connect layers after data checks
        self._connect_layers()

        # set compiler flag
        self._compile_flag = True

    def fit(self, batch_size : int, epochs : int, logging : bool):

        # for logging
        net_loss = 0.0

        # Check compile flag
        if self._compile_flag:
            for _ in tqdm(range(epochs), desc="Training progress bar (epochs)"):
                self._rand_data()
                mini_batches = self._group_batches(batch_size)

                for i, (x_batch, y_batch) in enumerate(mini_batches):
                    yhat = self._forward(x_batch, training=True)
                    self._backward(y_batch) # Set error terms for each layer

                    if logging: net_loss += self._loss(yhat, y_batch)

                    parameters = self._get_parameters()
                    gradients = self._get_gradients()

                    if isinstance(self._optimizer, SGD):
                        self._optimizer.step(parameters, gradients)
                    elif isinstance(self._optimizer, (SGDWithMomentum, RMSProp, AdaptiveMovementEstimation)):
                        self._optimizer.step(parameters, gradients, i + 1)

        if logging: return net_loss
        else: return None

    def _loss(self, yhat, y):
        reg_loss = sum(layer.reg_loss() for layer in self._layers if layer.l_type == "dense")

        return self._data_loss(yhat, y) + reg_loss

    def _data_loss(self, yhat, y):

        m = y.shape[0]
        if self._loss == MSE:
            loss = np.sum((yhat - y) ** 2) / m
        elif self._loss == BCE:
            true_class_indices = np.argmax(y, axis=1)
            log_probs = np.log(yhat[np.arange(m), true_class_indices] + 1e-10)  # avoid log(0)
            loss = -np.sum(log_probs) / m
        elif self._loss == SCCE:
            """
            SCCE can get confusing so I added some comments
            """

            # Prevent log(0) error
            epsilon = 1e-12

            # Set bounds of predictions using np.clip — method that truncates values
            yhat = np.clip(yhat, epsilon, 1. - epsilon)

            # Get true class labels (looks like [0, 0, 1, 0, 0, ...])
            true_class_indices = np.argmax(y, axis=1)

            # Find the probability associated with the true labels and log them
            log_probs = np.log(yhat[np.arange(m), true_class_indices])

            # Find sum of log probs and divide by example count to get average
            loss = -np.sum(log_probs) / m
        else:
            raise ValueError(f"Loss function '{self._loss}' is not implemented.")

        return loss

    def _get_parameters(self):
        params = []
        for layer in self._layers[1:]:
            if hasattr(layer, 'w_m') and layer.w_m is not None and hasattr(layer, 'b_v') and layer.b_v is not None:
                params.append((layer.w_m, layer.b_v))
        return params

    def _get_gradients(self):
        return [layer.gradient for layer in self._layers[1:] if hasattr(layer, 'gradient') and layer.gradient is not None]

    def _rand_data(self):

        rand_indices = np.arange(len(self._x))
        np.random.shuffle(rand_indices)

        self._x = self._x[rand_indices]
        self._y = self._y[rand_indices]

        return self._x, self._y

    def _group_batches(self, batch_size : int):

        batches = []
        for i in range(0, len(self._x), batch_size):

            feature_batch = self._x[ i : i + batch_size]
            label_batch = self._y[ i : i + batch_size]

            batches.append((feature_batch, label_batch))

        return batches

    def _forward(self, x, training=True):

        # for layer in self._layers:
        #     # Configure DropOutLayer for training
        #     if isinstance(layer, DropOutLayer):
        #         layer._training = training

        return self._layers[0].forward_pass(x)

    def _backward(self, label):

        self._layers[-1].back_pass(
            label=label,
            loss=self._loss,
            layer_type="output"
        )

    def predict(self, x):
        self._forward(x, training=False)
        return self._layers[0].forward_pass(x)