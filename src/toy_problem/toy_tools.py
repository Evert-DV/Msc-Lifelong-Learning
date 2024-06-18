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
    def __init__(self, state_size=2, **kwargs):
        super().__init__(**kwargs)
        self.state_size = state_size
        self.dense1 = layers.Dense(32, activation='sigmoid')
        self.dense2 = layers.Dense(32, activation='leaky_relu')
        self.dense3 = layers.Dense(self.state_size)

        self.regularizer = keras.Sequential([
            layers.Lambda(lambda x: x)  # dummy pass-through
        ])

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        target = self.dense3(x)

        reg_input = ops.concatenate([inputs[..., :self.state_size], inputs[..., self.state_size:] + target], axis=-1)
        self.regularizer(reg_input)

        # target = layers.Lambda(lambda x: x[..., -self.state_size:])(reg_out)

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


def prep_data(data, prediction_window, state_size=2, interval=15, freq=50, val_split=None):
    # First sort by target
    # windowed_data = ops.array(
    #     [data[i:i + interval * freq] for i in range(0, len(data) - interval * freq + 1)[::interval * freq]])
    windowed_data = [data[np.all(data[..., -state_size:] == i, axis=-1)] for i in
                     np.unique(data[..., -state_size:], axis=0)]
    features = ops.concatenate(
        ops.concatenate((array[..., :-prediction_window, :state_size], array[..., prediction_window:, :state_size]),
                        axis=-1) for array in windowed_data)
    # features = ops.concatenate(
    #     (windowed_data[..., :-prediction_window, list(range(state_size))],
    #      windowed_data[..., prediction_window:, list(range(state_size))]),
    #     axis=-1).reshape(-1, 2 * state_size)  # state transitions
    # labels = ops.array([windowed_data[j, i:i + prediction_window, 2] for j in range(windowed_data.shape[0]) for i in
    #                     range(windowed_data.shape[1] - prediction_window)])  # control actions as labels
    labels = ops.concatenate(
        (array[..., prediction_window:, -state_size:] - array[..., prediction_window:, :state_size]) for array in
        windowed_data)
    # labels = ops.array(
    #     [windowed_data[j, i + prediction_window, -state_size:] - windowed_data[j, i + prediction_window,
    #                                                              -2 * state_size:-state_size] for j in
    #      range(windowed_data.shape[0]) for i in
    #      range(windowed_data.shape[1] - prediction_window)])  # target - future states as labels

    if val_split is not None:
        dataset = TensorDataset(features, labels)
        train_size = int((1 - val_split) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        return train_dataset, val_dataset
    else:
        return features, labels
