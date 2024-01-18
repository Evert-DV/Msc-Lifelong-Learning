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

    def compute_control(self, current_state, target, dt):
        position_error = target[0] - current_state[0]
        velocity_error = target[1] - current_state[1]
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


def simple_ilc(response, target, gain=5.):
    error = target - response
    target += gain * error
    return target


def main():
    seed = np.random.randint(0, 1000)
    np.random.seed(seed)
    print(f"Seed: {seed}")
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
    adapted_targets = []
    default_controls = []
    adapted_controls = []
    predictions = []
    labels = []
    t = np.arange(0, 240, dt)
    target = np.array([11., 0.])
    buffer = []
    x0_naive = x0
    adapted_target = np.copy(target)

    for ti in t:
        # if ti % 30 == 0 and ti != 0:
        #     print("\nFitting model...")
        #     adapter.train()
        #     buffer = np.asarray(buffer)
        #     features = buffer[]  # concat with current action
        #     label_action = buffer[1:, 2:3]  # next control actions as labels
        #     features = torch.from_numpy(features).float()
        #     features.requires_grad = True
        #     label_action = torch.from_numpy(label_action).float()
        #
        #     for epoch in range(100):
        #         optimizer.zero_grad()
        #         output = adapter(features)
        #         loss = loss_fn(output, label_action)
        #         loss.backward()
        #         optimizer.step()
        #         print(f"\rEpoch {epoch}\t Loss: {loss:.2f}", end="")
        #
        #     buffer = []

        if ti % 20 == 0:
            target = [np.random.rand() * 6 + 7, 0.]

        # adapter.eval()
        targets.append(target)
        control_action = controller.compute_control(x0, adapted_target, dt)
        a_naive = controller.compute_control(x0_naive, target, dt)
        default_controls.append(a_naive)
        adapted_controls.append(control_action)

        x = system.response(x0, control_action, do_update=False)
        x_naive = system.response(x0_naive, a_naive, do_update=False)

        adapted_target = simple_ilc(x, target)
        adapted_targets.append(adapted_target)

        signal.append(x)
        signal_naive.append(x_naive)
        buffer.append([*x0, control_action, *x, *target])

        x0 = x
        x0_naive = x_naive

        # if ti % 15 == 0:
        #     x0 = [9.9 + (np.random.rand() - .5) * .5, 0]
        #     x0_naive = x0

    signal = np.asarray(signal)
    signal_naive = np.asarray(signal_naive)
    targets = np.asarray(targets)
    adapted_targets = np.asarray(adapted_targets)

    fig, ax = plt.subplots(2, 1, sharex=True)

    ax[0].plot(t, signal_naive[:, 0], label="Default controller")
    ax[0].plot(t, signal[:, 0], label="Adaptive controller")
    ax[0].plot(t, targets[:, 0], '--', label="Target position")
    ax[0].plot(t, adapted_targets[:, 0], '--', label="Adapted target position")
    ax[0].invert_yaxis()
    ax[0].legend()

    # ax[1].plot(t, predictions, label="Predicted control actions")
    ax[1].plot(t, default_controls, label="Default control actions")
    ax[1].plot(t, adapted_controls, label="Adapted control actions")
    ax[1].legend()

    fig.tight_layout()
    if not os.path.exists("./tmp"):
        os.makedirs("./tmp")
    fig.savefig("./tmp/plot.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    if torch.cuda.is_available():
        with torch.cuda.device(0):
            main()
    else:
        main()
