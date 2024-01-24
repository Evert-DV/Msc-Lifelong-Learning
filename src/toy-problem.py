import os

os.environ["KERAS_BACKEND"] = "torch"
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import torch
import keras
from keras import layers
from keras.models import load_model, save_model
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy


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


class Adapter(keras.Sequential):
    def __init__(self, input_size, output_size):
        super(Adapter, self).__init__()
        self.add(layers.Dense(32, activation='softmax'))
        self.add(layers.Dense(32, activation='leaky_relu'))
        self.add(layers.Dense(output_size))


def main():
    pretrain = False
    seed = np.random.randint(0, 1000)
    print(f"Seed: {seed}")
    np.random.seed(seed)
    if pretrain:
        np.random.seed(16)

    system = System(5, 10, 3, 5)
    controller = PIDController(350, 107.5, 1257)
    reference_controller = PIDController(350, 107.5, 1257)

    prediction_window = 10
    # adapter = Adapter(4, prediction_window)
    adapter = Adapter(4, 2)
    if not pretrain:
        adapter = load_model('./tmp/target_adapter.keras')
        # pass
    optimizer = keras.optimizers.Adam(learning_rate=1.e-2)
    loss_fn = keras.losses.MeanSquaredError()
    adapter.compile(optimizer=optimizer, loss=loss_fn)

    if pretrain:
        pretrain_data = np.load("./tmp/pretrain_data.npy")

        print("\nPretraining model...")
        # features = np.concatenate(
        #     (pretrain_data[:-prediction_window, [0, 1]], pretrain_data[prediction_window:, [3, 4]]),
        #     axis=1)  # state transitions
        # labels = np.asarray(
        #     [pretrain_data[i:i + prediction_window, 2:3].ravel() for i in
        #      range(len(pretrain_data) - prediction_window)])  # control actions as labels
        features = pretrain_data[:, [0, 1, 3, 4]]
        labels = pretrain_data[:, -2:]
        dataset = TensorDataset(torch.from_numpy(features), torch.from_numpy(labels))
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

        callback = keras.callbacks.EarlyStopping(monitor='loss',
                                                 mode='min',
                                                 min_delta=1e-4,
                                                 patience=5)

        history = adapter.fit(dataloader,
                              epochs=100,
                              batch_size=32,
                              callbacks=[callback])

        save_model(adapter, './tmp/target_adapter.keras')

    dt = 1 / 60
    x0 = [9.9, 0]  # 9.9 was found to be the steady state
    signal = []
    signal_naive = []
    targets = []
    reference_controls = []
    adapted_controls = []
    adapted_targets = []
    predicted_targets = []
    t = np.arange(0, 240, dt)
    target = np.array([11., 0.])
    buffer = []
    x0_naive = [9.9, 0]

    if not pretrain:
        for ti in t:
            # if ti % 15 == 0 and ti != 0:
            #     print("\nFitting model...")
            #     adapter.train()
            #     buffer = np.asarray(buffer)
            #     # features = np.concatenate((buffer[:-prediction_window, [0, 1]], buffer[prediction_window:, [3, 4]]),
            #     #                           axis=1)  # state transitions
            #     # labels = np.asarray(
            #     #     [buffer[i:i + prediction_window, 2:3].ravel() for i in
            #     #      range(len(buffer) - prediction_window)])  # control actions as labels
            #     features = buffer[:, [0, 1, 3, 4]]
            #     labels = buffer[:, -2:]
            #     dataset = TensorDataset(torch.from_numpy(features).float(), torch.from_numpy(labels).float())
            #     X, y = dataset.tensors
            #
            #     for epoch in range(100):
            #         optimizer.zero_grad()
            #         output = adapter(X)
            #         loss = loss_fn(output, y)
            #         loss.backward()
            #         optimizer.step()
            #         print(f"\rEpoch {epoch}\t Loss: {loss:.2f}", end="")
            #
            #     buffer = []

            if ti % 15 == 0:
                target = [np.random.rand() * 6 + 7, 0.]

            if ti < 0.:  # essentially disable adapter
                adapted_target = adapter(torch.tensor([control_action, *target]).float())
                adapted_target = adapted_target.detach().numpy()
                adapted_targets.append(adapted_target)
            else:
                adapted_target = target
                adapted_targets.append(2 * [float('nan')])

            targets.append(target)
            control_action = controller.compute_control(x0, adapted_target, dt)

            # if ti > 15:
            #     predicted_action = adapter(torch.tensor([*x0, *target]).float())
            #     adjustment = predicted_action[0].item() - control_action
            #     predicted_controls.append(predicted_action[0].item())
            # else:
            #     adjustment = 0
            #     predicted_controls.append(np.nan)

            # control_action += adjustment

            a_naive = reference_controller.compute_control(x0_naive, target, dt)
            reference_controls.append(a_naive)

            x = system.response(x0, control_action, do_update=False)
            x_naive = system.response(x0_naive, a_naive, do_update=False)

            prediction = adapter([*x0, *target])
            predicted_targets.append(prediction)

            adapted_controls.append(control_action)

            signal.append(x)
            signal_naive.append(x_naive)
            buffer.append([*x0, control_action, *x, *target])

            x0 = x
            x0_naive = x_naive

        signal = np.asarray(signal)
        signal_naive = np.asarray(signal_naive)
        targets = np.asarray(targets)
        predicted_targets = np.asarray(predicted_targets)
        adapted_targets = np.asarray(adapted_targets)

        fig, ax = plt.subplots(3, 1, sharex=True)

        ax[0].plot(t, signal[:, 0], label="Adaptive controller")
        ax[0].plot(t, signal_naive[:, 0], label="Default controller")
        ax[0].plot(t, targets[:, 0], '--', label="Target position")
        ax[0].invert_yaxis()
        ax[0].legend()

        ax[1].plot(t, adapted_controls, label="Adapted control actions")
        ax[1].plot(t, reference_controls, '--', label="Default control actions")
        ax[1].legend()

        ax[2].plot(t, targets[:, 0], label="Reference targets")
        ax[2].plot(t, predicted_targets[:, 0], '--', label="Predicted targets")
        ax[2].plot(t, adapted_targets[:, 0], label="Adapted targets")
        ax[2].invert_yaxis()
        ax[2].legend()

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
