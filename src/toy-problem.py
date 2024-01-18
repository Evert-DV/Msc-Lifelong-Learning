import os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


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


def main():
    pretrain = False
    seed = np.random.randint(0, 1000)
    print(f"Seed: {seed}")
    np.random.seed(seed)
    if pretrain:
        np.random.seed(16)

    torch.manual_seed(16)

    system = System(5, 10, 3, 5)
    controller = PIDController(350, 107.5, 1257)

    adapter = Adapter(4, 10)
    if not pretrain:
        adapter.load_state_dict(torch.load('./tmp/adapter_state_dict.pth'))
        # pass
    optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    if pretrain:
        pretrain_data = np.load("./tmp/pretrain_data.npy")

        print("\nPretraining model...")
        adapter.train()
        features = np.concatenate((pretrain_data[:-10, [0, 1]], pretrain_data[10:, [3, 4]]),
                                  axis=1)  # state transitions
        label_action = np.asarray(
            [pretrain_data[i:i + 10, 2:3].ravel() for i in range(len(pretrain_data) - 10)])  # control actions as labels
        dataset = TensorDataset(torch.from_numpy(features).float(), torch.from_numpy(label_action).float())
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

        for epoch in range(100):
            for inputs, targets in dataloader:
                optimizer.zero_grad()
                output = adapter(inputs)
                loss = loss_fn(output, targets)
                loss.backward()
                optimizer.step()
            print(f"\rEpoch {epoch}\t Loss: {loss.item():.2f}", end="")

        torch.save(adapter.state_dict(), './tmp/adapter_state_dict.pth')

    dt = 1 / 60
    x0 = [9.9, 0]  # 9.9 was found to be the steady state
    signal = []
    signal_naive = []
    targets = []
    default_controls = []
    adapted_controls = []
    predicted_controls = []
    t = np.arange(0, 600, dt)
    target = np.array([11., 0.])
    buffer = []
    x0_naive = [9.9, 0]

    if not pretrain:
        for ti in t:
            if ti % 15 == 0 and ti != 0:
                print("\nFitting model...")
                adapter.train()
                buffer = np.asarray(buffer)
                features = np.concatenate((buffer[:-10, [0, 1]], buffer[10:, [3, 4]]), axis=1)  # state transitions
                label_action = np.asarray(
                    [buffer[i:i + 10, 2:3].ravel() for i in range(len(buffer) - 10)])  # control actions as labels
                dataset = TensorDataset(torch.from_numpy(features).float(), torch.from_numpy(label_action).float())
                X, y = dataset.tensors

                for epoch in range(100):
                    optimizer.zero_grad()
                    output = adapter(X)
                    loss = loss_fn(output, y)
                    loss.backward()
                    optimizer.step()
                    print(f"\rEpoch {epoch}\t Loss: {loss:.2f}", end="")

                buffer = []

            if ti % 20 == 0:
                target = [np.random.rand() * 6 + 7, 0.]

            adapter.eval()
            targets.append(target)
            control_action = controller.compute_control(x0, target, dt)

            if ti > 15:
                predicted_action = adapter(torch.tensor([*x0, *target]).float())
                adjustment = predicted_action[0].item() - control_action
                predicted_controls.append(predicted_action[0].item())
            else:
                adjustment = 0
                predicted_controls.append(0)

            control_action += adjustment

            a_naive = controller.compute_control(x0_naive, target, dt)
            default_controls.append(a_naive)

            x = system.response(x0, control_action, do_update=True)
            x_naive = system.response(x0_naive, a_naive, do_update=False)

            adapted_controls.append(control_action)

            signal.append(x)
            signal_naive.append(x_naive)
            buffer.append([*x0, control_action, *x, *target])

            x0 = x
            x0_naive = x_naive

        signal = np.asarray(signal)
        signal_naive = np.asarray(signal_naive)
        targets = np.asarray(targets)
        labels = default_controls

        fig, ax = plt.subplots(3, 1, sharex=True)

        ax[0].plot(t, signal[:, 0], label="Adaptive controller")
        ax[0].plot(t, signal_naive[:, 0], label="Default controller")
        ax[0].plot(t, targets[:, 0], '--', label="Target position")
        ax[0].invert_yaxis()
        ax[0].legend()

        ax[1].plot(t, adapted_controls, label="Adapted control actions")
        ax[1].plot(t, labels, '--', label="Default control actions")
        ax[1].legend()

        ax[2].plot(t, predicted_controls, label="Predicted control actions")
        ax[2].plot(t, adapted_controls, '--', label="Control actions (labels)")

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
