import os

os.environ["KERAS_BACKEND"] = "torch"
import numpy as np
import scipy as sp
import keras
from keras import ops
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
        self.k *= 0.995
        self.c *= 0.995
        self.l0 += 1 / self.l0
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
        position_error = target[0] - current_state[0]
        velocity_error = target[1] - current_state[1]
        self.integral_error += position_error * dt

        control_action = self.kp * position_error + self.kd * velocity_error + self.ki * self.integral_error

        return control_action


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


def prep_data(data, prediction_window, interval=15, val_split=None):
    # First sort by target
    windowed_data = ops.array(
        [data[i:i + interval * 60] for i in range(0, len(data) - interval * 60 + 1)[::interval * 60]])
    features = ops.concatenate(
        (windowed_data[..., :-prediction_window, [0, 1]], windowed_data[..., prediction_window:, [0, 1]]),
        axis=-1).reshape(-1, 4)  # state transitions
    labels = ops.array([windowed_data[j, i:i + prediction_window, 2] for j in range(windowed_data.shape[0]) for i in
                        range(windowed_data.shape[1] - prediction_window)])  # control actions as labels

    if val_split is not None:
        dataset = TensorDataset(features, labels)
        train_size = int((1 - val_split) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        return train_dataset, val_dataset
    else:
        return features, labels


def simple_ilc(response, target, previous_error, gain=.5):
    error = ops.convert_to_numpy(target - response)
    gain = adaptive_gain(np.linalg.norm(error), previous_error, gain)
    adaptations = gain * error
    return adaptations, gain, np.linalg.norm(error)


def predictive_ilc(response, target, previous_error, gain=.5, step=5):
    future_errors_sum = []
    for i in range(len(target)):
        future_errors_sum.append(np.sum(target[i:i + step] - response[i:i + step], axis=0))
    gain = adaptive_gain(np.linalg.norm(future_errors_sum), previous_error, gain)
    adaptations = gain * np.asarray(future_errors_sum)
    return adaptations, gain, np.linalg.norm(future_errors_sum)


def adaptive_gain(error, previous_error, gain, max_gain=5.):
    gain *= 1.1 if error < previous_error else 0.9
    return min(gain, max_gain)


def ilc_loss_fn(errors):
    pos_loss = torch.mean(torch.abs(errors[::2]))
    vel_loss = torch.mean(torch.abs(errors[1::2]))
    return 1. * pos_loss + 1. * vel_loss


def ilc_nn(response, target, model, optimizer, old_errors, old_adaptations, update_ilc=True):
    model.train()

    # the errors of the past 15 sec, after the previous update
    errors = (response - target).ravel()
    errors.requires_grad = True
    errors.retain_grad()

    # the gains used during the past 15 sec
    gains = model(ops.ones(1)[None])[0]
    gains.retain_grad()

    delta_pre_prev_update = old_adaptations.ravel()  # adaptations before the past 15 sec, before the previous update
    delta_post_prev_update = old_errors * gains  # adaptations of the past 15 sec, after the previous update
    delta_post_prev_update.retain_grad()

    if update_ilc:
        model.train()
        optimizer.zero_grad()
        # Loss is calculated as mse of errors. dLoss/dGain = dLoss/dErrors * dErrors/dDelta * dDelta/dGain
        loss = ilc_loss_fn(errors)
        loss.backward()
        dl_de = errors.grad
        de_delta = (errors - old_errors) / (delta_post_prev_update - delta_pre_prev_update + 1e-32)
        # From here, autograd should take care of the rest
        delta_post_prev_update.backward(dl_de * de_delta)
        optimizer.step()
        print(f"Loss: {ilc_loss_fn(errors).item():.3f}\n"
              f"Parameters have gradients: {not (ops.isnan(model.trainable_weights[0].value.grad).any()).item()}")

    model.eval()
    new_gains = model(ops.ones(1)[None])[0]
    new_delta = new_gains * errors  # new adaptations for the coming 15 sec

    return ops.reshape(new_delta, (15 * 60, 2)), delta_post_prev_update, errors


def train_iterative(model, optimizer, loss_fn, target, response, old_output, old_errors):
    model.train()

    # the errors of the past iteration, after the previous update
    errors = (response - target).ravel()
    errors.requires_grad = True
    errors.retain_grad()

    model_output = model(ops.ones(1)[None])[0]
    model_output.retain_grad()

    # The train step
    optimizer.zero_grad()
    loss = loss_fn(errors)
    loss.backward()
    d_errors_d_output = (errors - old_errors.ravel()) / (model_output - old_output + 1e-32)
    model_output.backward(d_errors_d_output * errors.grad)
    optimizer.step()

    print(f"Loss: {ilc_loss_fn(errors).item():.3f}\n"
          f"Parameters have gradients: {not (ops.isnan(model.trainable_weights[0].value.grad).any()).item()}")

    return model_output, errors
