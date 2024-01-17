import os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


# Create a mass-spring-damper toy problem
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

    def compute_control(self, current_state, target_pos, dt):
        position_error = target_pos - current_state[0]
        velocity_error = -current_state[1]
        self.integral_error += position_error * dt

        control_action = self.kp * position_error + self.kd * velocity_error + self.ki * self.integral_error

        return control_action


class Adapter(nn.Sequential):
    def __init__(self, input_size, output_size):
        super(Adapter, self).__init__(
            nn.Linear(input_size, 32),
            nn.Softsign(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.Linear(32, output_size),
        )


def main():
    np.random.seed(16)
    torch.manual_seed(16)

    system = System(5, 10, 3, 5)
    controller = PIDController(350, 107.5, 1257)
    adapter = Adapter(2, 1)

    optimizer = torch.optim.SGD(adapter.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    dt = 1 / 60
    x0 = [9.9, 0]  # 9.9 was found to be the steady state
    signal = []
    signal_naive = []
    targets = []
    controls = []
    predictions = []
    labels = []
    t = np.arange(0, 300, dt)
    target = np.array([11.])
    buffer = []
    x0_naive = x0
    predicted_action = [0.]

    for ti in t:
        if ti % 30 == 0 and ti != 0:
            print("\nFitting model...")
            adapter.train()
            buffer = np.asarray(buffer)
            error_labels = buffer[:-1, 0:1] - buffer[:-1, -1:]  # current error as labels
            features = np.concatenate((error_labels, buffer[:-1, 2:3]), axis=1)  # concat with current action
            # label_e = buffer[:, -3:-2] - buffer[:, -1:]
            label_action = buffer[1:, 2:3]  # next control actions as labels
            features = torch.from_numpy(features).float()
            features.requires_grad = True
            label_action = torch.from_numpy(label_action).float()

            for epoch in range(100):
                optimizer.zero_grad()
                output = adapter(features)
                loss = loss_fn(output, label_action)
                loss.backward()
                optimizer.step()
                print(f"\rEpoch {epoch}\t Loss: {loss:.2f}", end="")

            buffer = []

        # if ti % 20 == 0:
        #     target = np.random.rand(1) * 6 + 7

        adapter.eval()
        targets.append(target)
        control_action = controller.compute_control(x0, target, dt)
        # control_action = control_action + predicted_action
        a_naive = controller.compute_control(x0_naive, target, dt)
        controls.append(control_action)

        disturbance = 0.
        # if np.random.rand() < 0.005:
        #     disturbance = np.random.rand(1) * 10000

        x = system.response(x0, control_action + disturbance, do_update=False)
        x_naive = system.response(x0_naive, a_naive + disturbance, do_update=False)
        # labels.append(x[0] - target)
        labels.append(control_action)
        predicted_action = adapter(torch.tensor([x0[0] - target[0], *control_action]).float())  # e = x0[0] - target[0]
        predictions.append(predicted_action.item())
        predicted_action = predicted_action.item()

        signal.append(x)
        signal_naive.append(x_naive)
        buffer.append([*x0, *control_action, *x, *target])

        x0 = x
        x0_naive = x_naive

        if ti % 15 == 0:
            x0 = [9.9 + (np.random.rand() - .5) * .5, 0]
            x0_naive = x0

    signal = np.asarray(signal)
    signal_naive = np.asarray(signal_naive)

    fig, ax = plt.subplots(3, 1, sharex=True)

    ax[0].plot(t, signal[:, 0], label="Adaptive controller")
    ax[0].plot(t, signal_naive[:, 0], label="Default controller")
    ax[0].plot(t, targets, '--', label="Target")
    ax[0].invert_yaxis()
    ax[0].legend()

    ax[1].plot(t, controls, label="Control actions (default)")
    ax[1].invert_yaxis()
    ax[1].legend()

    ax[2].plot(t[1:], predictions[:-1], label="Predicted control actions")
    ax[2].plot(t[1:], labels[1:], '--', label="Actual control actions")
    ax[2].legend()

    fig.tight_layout()
    if not os.path.exists("./tmp"):
        os.makedirs("./tmp")
    # fig.savefig("./tmp/plot.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    if torch.cuda.is_available():
        with torch.cuda.device(0):
            main()
    else:
        main()
