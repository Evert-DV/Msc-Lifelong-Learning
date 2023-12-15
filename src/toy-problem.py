import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


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
    def __init__(self, kp, kd, ki, dt):
        self.kp = kp
        self.kd = kd
        self.ki = ki
        self.integral_error = 0.
        self.dt = dt

    def compute_control(self, current_state, target_pos):
        position_error = target_pos - current_state[0]
        velocity_error = -current_state[1]
        self.integral_error += position_error * self.dt

        control_action = self.kp * position_error + self.kd * velocity_error + self.ki * self.integral_error
        # control_action = np.maximum(-250, np.minimum(250, control_action))

        return control_action


class Adapter(nn.Module):
    def __init__(self, controller, system):
        super(Adapter, self).__init__()
        self.state_adjuster = nn.Sequential(
            nn.Linear(2, 8),
            nn.Linear(8, 2))
        # Define action adjuster layers
        self.action_adjuster = nn.Sequential(
            nn.Linear(1, 8),
            nn.Linear(8, 1))
        # Controller and system functions
        self.controller = controller
        self.system = system

    def forward(self, inputs):
        results = []
        for input in inputs:
            # Extract state, action, target from the input tensor
            state = input[:2]         # Convert to NumPy array or list
            action = input[2].item()  # Extract single value as Python float
            target = input[3].item()  # Extract single value as Python float

            # Pass them to the controller and system as regular Python types
            adjusted_state = self.state_adjuster(state)

            adjusted_state_np = adjusted_state.detach().numpy()
            adjusted_action = self.controller.compute_control(adjusted_state_np, target)

            adjusted_action_tensor = torch.tensor([adjusted_action], dtype=torch.float32)
            adjusted_action_tensor = self.action_adjuster(adjusted_action_tensor)

            adjusted_action_np = adjusted_action_tensor.item()
            result = self.system.response(adjusted_state_np, adjusted_action_np)

            # Convert the result back to a tensor and store
            results.append(torch.tensor(result[0]))

        # Stack all results into a single tensor
        return torch.stack(results)


def main():
    np.random.seed(16)

    dt = 1 / 60

    system = System(5, 10, 3, 5)
    controller = PIDController(350, 107.5, 1257, dt)
    adapter = Adapter(controller, system)
    # Define optimizer and loss function
    optimizer = optim.SGD(adapter.parameters(), lr=0.01)
    loss_function = torch.nn.MSELoss()

    x0 = [9.9, 0]  # 9.9 was found to be the steady state
    signal = []
    targets = []
    controls = []
    t = np.arange(0, 45, dt)
    target = None
    buffer = []

    for ti in t:
        if ti % 10 == 0 and ti != 0.:
            print("fitting")
            # Convert buffer to a PyTorch tensor
            buffer_tensor = torch.tensor(buffer, dtype=torch.float32)
            # Training loop
            for epoch in range(10):  # Define num_epochs as needed
                optimizer.zero_grad()
                output = adapter(buffer_tensor)
                loss = loss_function(output, buffer_tensor[:, -1])
                loss.backward()
                optimizer.step()
        if ti % 15 == 0:
            target = np.random.rand(1) * 6 + 7
        targets.append(target)
        a = controller.compute_control(x0, target)
        controls.append(a)
        x = system.response(x0, a, do_update=True)
        signal.append(x)
        buffer.append([*x0, *a, *x, *target])
        x0 = x

    signal = np.asarray(signal)

    fig, ax = plt.subplots(2, 1, sharex=True)

    ax[0].plot(t, signal[:, 0])
    ax[0].plot(t, targets)
    ax[0].invert_yaxis()

    ax[1].plot(t, controls)
    ax[1].invert_yaxis()

    fig.tight_layout()
    plt.savefig("./tmp/plot.png", dpi=300)

    # # System identification
    # z = c / (2 * np.sqrt(m * k))
    # wn = np.sqrt(k / m) * np.sqrt(1 - z ** 2)


if __name__ == "__main__":
    main()
