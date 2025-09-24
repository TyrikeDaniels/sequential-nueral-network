"""
This file serves as the basis of creating optimizers to use in the
sequential neural network. Supporting optimizers include:
- Stochastic gradient descent (SGD)
- Root mean squared propagation (RMSProp)
- Stochastic gradient descent with momentum (SGDM or SGDWithMomentum)
- Adaptive movement estimation (AdaM).
"""


class SGD:
    def __init__(self, lr : float):
        self._lr = lr

    def step(self, parameters, gradients):
        for (weights, bias), grad in zip(parameters, gradients):
            weights -= self._lr * grad["weights"]
            bias -= self._lr * grad["biases"]


class RMSProp:
    def __init__(self, learning_rate=0.001, decay=0.9, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.v_t = {} # second movement

    def step(self, parameters, gradients, t):
        for (weights, bias), grad in zip(parameters, gradients):
            param_id = id(weights) # acts as unique identifier

            if param_id not in self.v_t:
                self.v_t[param_id] = {"weights": 0, "biases": 0}

                self.v_t[param_id]["weights"] = (
                        self.decay * self.v_t[param_id]["weights"] + (1 - self.decay) * grad["weights"] ** 2
                )
                self.v_t[param_id]["biases"] = (
                        self.decay * self.v_t[param_id]["biases"] + (1 - self.decay) * grad["biases"] ** 2
                )

                # Bias correction: Correct for the fact that m_t is biased toward zero initially
                v_w_corrected = self.v_t[param_id]["weights"] / (1 - self.decay ** t)
                v_b_corrected = self.v_t[param_id]["biases"] / (1 - self.decay ** t)

                # Update parameters (weights and biases) using the corrected squared gradient
                weights -= self.learning_rate * grad["weights"] / (v_w_corrected ** 0.5 + self.epsilon)
                bias -= self.learning_rate * grad["biases"] / (v_b_corrected ** 0.5 + self.epsilon)

class SGDWithMomentum:
    def __init__(self, learning_rate, decay=0.9):
        self.learning_rate = learning_rate
        self.decay = decay
        self.m_t = {} # first movement

    def step(self, parameters, gradients, t):
        for (weights, bias), grad in zip(parameters, gradients):
            param_id = id(weights)  # Unique identifier for the current parameter

            if param_id not in self.m_t:
                self.m_t[param_id] = {"weights": 0, "biases": 0}

            # Update the momentum term for weights and biases using the gradient
            self.m_t[param_id]["weights"] = (
                    self.decay * self.m_t[param_id]["weights"] + (1 - self.decay) * grad["weights"]
            )
            self.m_t[param_id]["biases"] = (
                    self.decay * self.m_t[param_id]["biases"] + (1 - self.decay) * grad["biases"]
            )

            # Bias correction
            m_w_corrected = self.m_t[param_id]["weights"] / (1 - self.decay ** t)
            m_b_corrected = self.m_t[param_id]["biases"] / (1 - self.decay ** t)

            # Update parameters (weights and biases) using the corrected momentum
            weights -= self.learning_rate * m_w_corrected
            bias -= self.learning_rate * m_b_corrected

class AdaptiveMovementEstimation:
    def __init__(self, learning_rate=0.001, decay_one=0.9, decay_two=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.decay_one = decay_one  # Decay for the first moment (m_t)
        self.decay_two = decay_two  # Decay for the second moment (v_t)
        self.epsilon = epsilon
        self.m_t = {}  # First moment (mean)
        self.v_t = {}  # Second moment (variance)

    def step(self, parameters, gradients, t):
        for (weights, bias), grad in zip(parameters, gradients):
            param_id = id(weights)  # Unique identifier for the current parameter

            if param_id not in self.m_t:
                # Initialize first and second moments for weights and biases
                self.m_t[param_id] = {"weights": 0, "biases": 0}
                self.v_t[param_id] = {"weights": 0, "biases": 0}

            # Update the first moment (m_t)
            self.m_t[param_id]["weights"] = (
                    self.decay_one * self.m_t[param_id]["weights"] + (1 - self.decay_one) * grad["weights"]
            )
            self.m_t[param_id]["biases"] = (
                    self.decay_one * self.m_t[param_id]["biases"] + (1 - self.decay_one) * grad["biases"]
            )

            # Update the second moment (v_t)
            self.v_t[param_id]["weights"] = (
                    self.decay_two * self.v_t[param_id]["weights"] + (1 - self.decay_two) * grad["weights"] ** 2
            )
            self.v_t[param_id]["biases"] = (
                    self.decay_two * self.v_t[param_id]["biases"] + (1 - self.decay_two) * grad["biases"] ** 2
            )

            # Bias correction for the first moment
            m_w_corrected = self.m_t[param_id]["weights"] / (1 - self.decay_one ** t)
            m_b_corrected = self.m_t[param_id]["biases"] / (1 - self.decay_one ** t)

            # Bias correction for the second moment
            v_w_corrected = self.v_t[param_id]["weights"] / (1 - self.decay_two ** t)
            v_b_corrected = self.v_t[param_id]["biases"] / (1 - self.decay_two ** t)

            # Update parameters using Adam update rule
            weights -= (self.learning_rate * m_w_corrected) / (v_w_corrected ** 0.5 + self.epsilon)
            bias -= (self.learning_rate * m_b_corrected) / (v_b_corrected ** 0.5 + self.epsilon)
