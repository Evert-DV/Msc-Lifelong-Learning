import numpy as np
import scipy as sp
import keras
from keras import ops, layers
import torch
from torch.utils.data import TensorDataset, random_split


class System:
    def __init__(self, m, k, c, l0=0, g=9.81):
        self.m = m
        self.k = k
        self.c = c
        self.l0 = l0
        self.g = g
        self.u = m * g + k * l0

        self.A = np.array([[0., 1.], [-k / m, -c / m]])
        self.B = np.array([[0., 1 / m]]).T
        self.C = np.array([[1., 0.]])
        self.D = np.array([[0.]])
        self.sys = sp.signal.StateSpace(self.A, self.B, self.C, self.D)

    def response(self, s, a=0, dt=0.01, do_update=False):
        _, _, s = sp.signal.lsim(self.sys, 2 * [self.u + a], [0, dt], s)

        if do_update:
            self.update()

        return s[-1]

    def update(self):
        self.k = max(0, self.k - 1e-4)
        self.c = max(0, self.c - 1e-4)
        self.l0 += 1 / (1000 * self.l0)
        self.u = self.m * self.g + self.k * self.l0

        self.A = np.array([[0., 1.], [-self.k / self.m, -self.c / self.m]])
        self.B = np.array([[0., 1 / self.m]]).T
        self.C = np.array([[1., 0.]])
        self.D = np.array([[0.]])
        self.sys = sp.signal.StateSpace(self.A, self.B, self.C, self.D)


class PIDController:
    def __init__(self, kp, kd, ki):
        self.kp = kp
        self.kd = kd
        self.ki = ki
        self.integral_error = 0.

    def compute_control(self, current_state, target, dt):
        if target is None:
            return 0
        else:
            position_error = target[0] - current_state[0]
            velocity_error = target[1] - current_state[1]
            self.integral_error += position_error * dt

            control_action = self.kp * position_error + self.kd * velocity_error + self.ki * self.integral_error

        return control_action


class TargetAdapter(keras.Model):
    def __init__(self, state_size=2, target_size=1, **kwargs):
        super().__init__(**kwargs)
        self.state_size = state_size
        self.target_size = target_size

        inputs = layers.Input(shape=(2 * state_size,))
        x = layers.Dense(32, activation='sigmoid')(inputs)
        x = layers.Dense(32, activation='leaky_relu')(x)
        y = layers.Dense(target_size)(x)
        self.adapter = keras.Model(inputs=inputs, outputs=y)

        self.regularizer = keras.Sequential([
            layers.Lambda(lambda x: x)  # dummy pass-through
        ])

    def call(self, inputs):
        target = self.adapter(inputs)

        reg_input = ops.concatenate(
            [inputs[..., :self.state_size], ops.pad(target, [[0, 0], [0, self.state_size - self.target_size]])],
            axis=-1)
        self.regularizer(reg_input)

        return target


class RMSERegularizer(layers.Layer):
    def __init__(self, state_size=2, weight=0.25, **kwargs):
        super().__init__(**kwargs)
        self.state_size = state_size
        self.weight = weight

    def call(self, inputs):
        y_true, y_pred = inputs[:, :self.state_size], inputs[:, self.state_size:]
        self.add_loss(self.weight * ops.mean((y_true - y_pred) ** 2) ** 0.5)
        return inputs


class EpochLogger(keras.callbacks.Callback):
    def __init__(self, verbose=1):
        super().__init__()
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if self.verbose:
            print(f"\rEpoch {epoch}\t", end="")

    def on_epoch_end(self, epoch, logs=None):
        if self.verbose:
            print(f"\r", end="")


def prep_data(data, prediction_window=None, state_size=2, target_size=1, true_target_list=None,
              val_split=None):
    window_list = prediction_window
    if type(prediction_window) is int:
        window_list = [prediction_window]
    elif prediction_window is None:
        window_list = [3, 5, 10, 15, 25]

    data = ops.array(data)
    # First sort by target
    if true_target_list is None:
        true_target_list = data[..., -target_size:].tolist()
    windowed_data = [data[ops.all(ops.array(true_target_list) == ops.array(i), axis=-1)] for i in
                     np.unique(true_target_list, axis=0)]
    features = ops.concatenate(
        ops.concatenate((array[..., :-window, :state_size], array[..., window:, :state_size]),
                        axis=-1) for array in windowed_data for window in window_list)
    labels = ops.concatenate(
        array[..., window:, -target_size:] for array in
        windowed_data for window in window_list)

    if val_split is not None:
        dataset = TensorDataset(features, labels)
        train_size = int((1 - val_split) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        return train_dataset, val_dataset
    else:
        return features, labels
