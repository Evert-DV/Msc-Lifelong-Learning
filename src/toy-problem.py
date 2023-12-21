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
    # torch.manual_seed(16)

    system = System(5, 10, 3, 5)
    controller = PIDController(350, 107.5, 1257)
    adapter = Adapter(3, 1)

    optimizer = torch.optim.SGD(adapter.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    dt = 1 / 60
    x0 = [9.9, 0]  # 9.9 was found to be the steady state
    signal = []
    targets = []
    controls = []
    t = np.arange(0, 60, dt)
    target = None
    buffer = []
    x0_adj = x0

    for ti in t:
        if ti % 10 == 0 and ti != 0:
            print("Fitting model...")
            adapter.train()
            buffer = np.asarray(buffer)
            features = buffer[:, :3]
            e = buffer[:, -3:-2] - buffer[:, -1:]
            features = torch.from_numpy(features).float()
            features.requires_grad = True
            e = torch.from_numpy(e).float()

            for epoch in range(100):
                print(f"\rEpoch {epoch}", end="")
                optimizer.zero_grad()
                output = adapter(features)
                loss = loss_fn(output, e)
                loss.backward()
                optimizer.step()

            buffer = []

        if ti % 120 == 0:
            target = np.random.rand(1) * 6 + 7

        adapter.eval()
        targets.append(target)
        a = controller.compute_control(x0_adj, target, dt)
        controls.append(a)
        x = system.response(x0, a, do_update=True)
        e = adapter(torch.tensor([*x0, *a]).float())
        x_adj = x
        x_adj[0] += e.item()

        signal.append(x)
        buffer.append([*x0, *a, *x, *target])

        x0 = x
        x0_adj = x_adj

    signal = np.asarray(signal)

    fig, ax = plt.subplots(2, 1, sharex=True)

    ax[0].plot(t, signal[:, 0])
    ax[0].plot(t, targets)
    ax[0].invert_yaxis()

    ax[1].plot(t, controls)
    ax[1].invert_yaxis()

    fig.tight_layout()
    if not os.path.exists("./tmp"):
        os.makedirs("./tmp")
    fig.savefig("./tmp/plot.png", dpi=300)


if __name__ == "__main__":
    if torch.cuda.is_available():
        with torch.cuda.device(0):
            main()
    else:
        main()
