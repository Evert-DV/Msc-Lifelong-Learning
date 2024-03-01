import numpy as np
import scipy as sp
import keras
from keras import ops
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
        position_error = target[0] - current_state[0]
        velocity_error = target[1] - current_state[1]
        self.integral_error += position_error * dt

        control_action = self.kp * position_error + self.kd * velocity_error + self.ki * self.integral_error

        return control_action


class FuzzyLogicController:
    def __init__(self):
        # Define fuzzy sets for the state variables and control action
        # For simplicity, using triangular membership functions
        self.fuzzy_sets = {
            'Low': lambda x: max(min((0.5 - x) / 0.5, 1), 0),
            'Medium': lambda x: max(min((x / 0.5, (1.0 - x) / 0.5)), 0),
            'High': lambda x: max(min((x - 0.5) / 0.5, 1), 0)
        }

    def fuzzify(self, value, fuzzy_set):
        # Calculate the degree of membership for a value in a fuzzy set
        return self.fuzzy_sets[fuzzy_set](value)

    @staticmethod
    def apply_rules(fuzzified_state, fuzzified_target):
        # Example fuzzy rules implementation
        control_action_fuzzy = np.zeros(3)  # Assuming three fuzzy sets for control action

        # Rule 1: If the current state is 'High', the control action is 'Low'
        control_action_fuzzy[0] += fuzzified_state[2]

        # Rule 2: If the current state is 'Low', the control action is 'High'
        control_action_fuzzy[2] += fuzzified_state[0]

        # More rules can be added based on system requirements

        return control_action_fuzzy

    @staticmethod
    def defuzzify(fuzzy_output):
        # Defuzzify using the centroid method
        # Assuming control action ranges from 0 to 1
        levels = np.array([0.25, 0.5, 0.75])  # Mid-points of 'Low', 'Medium', and 'High'
        return np.dot(fuzzy_output, levels) / sum(fuzzy_output) if sum(fuzzy_output) != 0 else 0

    def compute_control_action(self, current_state, target_state):
        # Fuzzify the current and target states
        fuzzified_current_state = np.array([self.fuzzify(current_state[0], fs) for fs in self.fuzzy_sets])
        fuzzified_target_state = np.array([self.fuzzify(target_state[0], fs) for fs in self.fuzzy_sets])

        # Apply fuzzy logic rules
        fuzzy_output = self.apply_rules(fuzzified_current_state, fuzzified_target_state)

        # Defuzzify to get the control action
        control_action = self.defuzzify(fuzzy_output)

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
    # labels = ops.array([windowed_data[j, i:i + prediction_window, 2] for j in range(windowed_data.shape[0]) for i in
    #                     range(windowed_data.shape[1] - prediction_window)])  # control actions as labels
    labels = ops.array([windowed_data[j, i + prediction_window, -2:] for j in range(windowed_data.shape[0]) for i in
                        range(windowed_data.shape[1] - prediction_window)])  # target states as labels

    if val_split is not None:
        dataset = TensorDataset(features, labels)
        train_size = int((1 - val_split) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        return train_dataset, val_dataset
    else:
        return features, labels
