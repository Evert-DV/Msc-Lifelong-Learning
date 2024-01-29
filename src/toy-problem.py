import os

os.environ["KERAS_BACKEND"] = "torch"
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import keras
from keras import ops, layers
from keras.saving import load_model
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split


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


def main():
    pretrain = False
    incremental_updates = True
    use_adaptation = True

    seed = np.random.randint(0, 1000)
    print(f"Seed: {seed}")
    np.random.seed(seed)
    if pretrain:
        torch.manual_seed(16)

    system = System(5, 10, 3, 5)
    controller = PIDController(350, 107.5, 1257)
    reference_controller = PIDController(350, 107.5, 1257)

    prediction_window = 10
    adapter = keras.Sequential([
        layers.Dense(32, activation='sigmoid'),
        layers.Dropout(0.25),
        layers.Dense(32, activation='leaky_relu'),
        layers.Dropout(0.25),
        layers.Dense(prediction_window)
    ])
    model_location = './tmp/action_adapter.keras'
    if not pretrain:
        adapter = load_model(model_location)

    optimizer = keras.optimizers.Adam(learning_rate=5.e-3)
    loss_fn = keras.losses.MeanSquaredError()
    adapter.compile(optimizer=optimizer, loss=loss_fn)

    if pretrain:
        pretrain_data = np.load("./tmp/pretrain_data.npy")
        train_set, val_set = prep_data(pretrain_data, prediction_window, interval=20, val_split=0.2)
        train_dataloader = DataLoader(train_set, batch_size=256, shuffle=True)
        val_dataloader = DataLoader(val_set, batch_size=256, shuffle=False)

        callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss',
                                                   mode='min',
                                                   min_delta=1e-5,
                                                   patience=10,
                                                   restore_best_weights=True,
                                                   verbose=1),
                     ]
        adapter.fit(train_dataloader,
                    epochs=100,
                    callbacks=callbacks,
                    validation_data=val_dataloader,
                    )

        adapter.save(model_location)

    dt = 1 / 60
    x0 = [9.9, 0]  # 9.9 was found to be the steady state
    signal = []
    reference_signal = []
    targets = []
    reference_controls = []
    adapted_controls = []
    predicted_controls = []
    t = np.arange(0, 240, dt)
    target = np.array([11., 0.])
    buffer = []
    x0_reference = [9.9, 0]

    if not pretrain:
        for ti in t:
            if ti % 15 == 0 and ti != 0:

                adapter.optimizer.lr = 1.e-3
                buffer = ops.array(buffer)
                features, labels = prep_data(buffer, prediction_window, interval=15)
                if incremental_updates:
                    print("\nFitting model...")
                    train_dataset, val_dataset = random_split(TensorDataset(features, labels),
                                                              [int(0.8 * len(features)),
                                                               len(features) - int(0.8 * len(features))])
                    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
                    val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=False)

                    callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss',
                                                               mode='min',
                                                               min_delta=1e-4,
                                                               patience=5,
                                                               restore_best_weights=True,
                                                               verbose=1),
                                 EpochLogger()
                                 ]
                    adapter.fit(train_dataloader,
                                epochs=100,
                                callbacks=callbacks,
                                validation_data=val_dataloader,
                                verbose=0,
                                )

                ref_prediction = adapter.predict(features, verbose=0)
                predicted_controls += ref_prediction[:, 0].ravel().tolist()
                predicted_controls += prediction_window * [float('nan')]

                buffer = []

            if ti % 15 == 0:
                target = [np.random.rand() * 6 + 7, 0.]

            targets.append(target)
            control_action = controller.compute_control(x0, target, dt)

            predicted_action = adapter.predict(ops.array([*x0, *target])[None], verbose=0)[0]
            adjustment = predicted_action[0] - control_action

            control_action += adjustment * use_adaptation

            reference_control = reference_controller.compute_control(x0_reference, target, dt)
            reference_controls.append(reference_control)

            x = system.response(x0, control_action, do_update=True)
            x_reference = system.response(x0_reference, reference_control, do_update=False)

            adapted_controls.append(control_action)

            signal.append(x)
            reference_signal.append(x_reference)
            buffer.append([*x0, control_action, *x, *target])

            x0 = x
            x0_reference = x_reference

        # adapter.save(model_location)

        signal = np.asarray(signal)
        reference_signal = np.asarray(reference_signal)
        targets = np.asarray(targets)
        predicted_controls = np.asarray(predicted_controls).ravel()
        adapted_controls = np.asarray(adapted_controls)

        fig, ax = plt.subplots(2, 1, sharex=True)

        ax[0].plot(t, signal[:, 0], label="Adaptive controller")
        ax[0].plot(t, reference_signal[:, 0], label="Default controller")
        ax[0].plot(t, targets[:, 0], '--', label="Target position")
        ax[0].invert_yaxis()
        ax[0].legend()

        ax[1].plot(t, adapted_controls, '-', label="Adapted control actions")
        ax[1].plot(t, reference_controls, label="Reference control actions")
        ax[1].plot(t[:-15 * 60], predicted_controls, ':', label="Predicted control actions")
        ax[1].legend()

        fig.tight_layout()
        if not os.path.exists("./tmp"):
            os.makedirs("./tmp")
        fig.savefig("./tmp/plot.png", dpi=300)
        plt.show()


if __name__ == "__main__":
    print("Using backend " + keras.backend.backend())
    if torch.cuda.is_available():
        print("Using CUDA")
        with torch.cuda.device(0):
            main()
    else:
        main()
