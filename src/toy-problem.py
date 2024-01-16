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
            nn.Linear(32, output_size),
            nn.ReLU()
        )


def main():
    np.random.seed(16)
    torch.manual_seed(16)

    system = System(5, 10, 3, 5)
    controller = PIDController(350, 107.5, 1257)
    adapter = Adapter(5, 1)

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
    x0_adj = x0
    x0_naive = x0

    for ti in t:
        if ti % 10 == 0 and ti != 0:
            print("\nFitting model...")
            adapter.train()
            buffer = np.asarray(buffer)
            features = buffer[:, :-1]
            label_e = buffer[:, -3:-2] - buffer[:, -1:]
            features = torch.from_numpy(features).float()
            features.requires_grad = True
            label_e = torch.from_numpy(label_e).float()

            for epoch in range(100):
                print(f"\rEpoch {epoch}", end="")
                optimizer.zero_grad()
                output = adapter(features)
                loss = loss_fn(output, label_e)
                loss.backward()
                optimizer.step()

            buffer = []

        # if ti % 20 == 0:
        #     target = np.random.rand(1) * 6 + 7

        adapter.eval()
        targets.append(target)
        a_w_adj = controller.compute_control(x0_adj, target, dt)
        a_naive = controller.compute_control(x0_naive, target, dt)
        controls.append(a_w_adj)

        disturbance = 0.
        # if np.random.rand() < 0.005:
        #     disturbance = np.random.rand(1) * 10000

        x = system.response(x0, a_w_adj + disturbance, do_update=False)
        x_naive = system.response(x0_naive, a_naive + disturbance, do_update=False)
        labels.append(x[0] - target)
        predicted_e = adapter(torch.tensor([*x0, *a_w_adj, *x]).float())
        predictions.append(predicted_e.item())
        x_adj = x
        # x_adj[0] -= predicted_e.item()

        signal.append(x)
        signal_naive.append(x_naive)
        buffer.append([*x0, *a_w_adj, *x, *target])

        x0 = x
        x0_naive = x_naive
        x0_adj = x_adj

        if ti % 30 == 0:
            x0 = [9.9, 0]
            x0_naive = x0
            # x0 = [np.random.rand() * 6 + 11., 0.]
            x0_adj = x0

    signal = np.asarray(signal)
    signal_naive = np.asarray(signal_naive)

    fig, ax = plt.subplots(3, 1, sharex=True)

    ax[0].plot(t, signal[:, 0])
    ax[0].plot(t, signal_naive[:, 0])
    ax[0].plot(t, targets, '--')
    ax[0].invert_yaxis()

    ax[1].plot(t, controls)
    ax[1].invert_yaxis()

    ax[2].plot(t, predictions)
    ax[2].plot(t, labels)

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
