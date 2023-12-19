import os
import copy
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
        self.k *= 0.999
        self.c *= 0.999
        self.l0 += 0.1 / self.l0
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


class CustomOperationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, custom_operation):
        # Store the custom_operation for later use in the backward pass
        ctx.custom_operation = custom_operation
        # Perform the forward pass using the provided custom_operation
        output = custom_operation(*inputs)
        ctx.save_for_backward(output, *inputs)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve the stored custom_operation
        custom_operation = ctx.custom_operation
        # Retrieve the saved tensors from the forward pass
        output, *inputs = ctx.saved_tensors

        # Use numerical differentiation to approximate the gradients for each input
        grad_inputs = []
        epsilon = 1e-6
        for input in inputs:
            grad_input = torch.autograd.functional.jacobian(custom_operation, input, create_graph=True)
            grad_input = grad_input.squeeze()  # Remove the extra dimension
            grad_inputs.append(grad_input)

        return (*grad_inputs, grad_output * 0), None  # Return gradients for all inputs (zeros for grad_output)


class Adapter(nn.Module):
    def __init__(self, controller, system):
        super(Adapter, self).__init__()
        # Define state adjuster layers
        self.state_adjuster = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
            nn.ReLU())
        # Define action adjuster layers
        self.action_adjuster = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.ReLU())
        # Controller and system functions
        self.controller = controller
        self.system = system

    def forward(self, inputs):
        controller, system = copy.deepcopy(self.controller.compute_control), copy.deepcopy(self.system.response)
        state, action, result, target = inputs[:, :2], inputs[:, 2:3], inputs[:, 3:5], inputs[:, 5:]

        adjusted_state = self.state_adjuster(state)

        actions = []
        for x, t in zip(adjusted_state, target):
            y = CustomOperationFunction.apply((x, t), controller)
            actions.append(torch.tensor(y, dtype=torch.float32))
        actions = torch.stack(actions)

        adjusted_action = self.action_adjuster(actions)

        results = []
        for x, a in zip(adjusted_state, adjusted_action):
            y = CustomOperationFunction.apply((x, a), system)
            results.append(torch.tensor(y))
        results = torch.stack(results)

        del controller, system
        return results


def main():
    np.random.seed(16)

    dt = 1 / 60

    system = System(5, 10, 3, 5)

    controller = PIDController(350, 107.5, 1257, dt)
    adapter = Adapter(controller, system)

    # Define optimizer and loss function
    optimizer = optim.SGD(adapter.parameters(), lr=0.1)
    loss_function = torch.nn.MSELoss()

    x0 = [9.9, 0]  # 9.9 was found to be the steady state
    signal = []
    targets = []
    controls = []
    t = np.arange(0, 60, dt)
    # target = None
    target = np.random.rand(1) * 6 + 7
    buffer = []
    losses = []

    for ti in t:
        if ti % 10 == 0 and ti != 0.:
            print("\nfitting")
            # training loop
            for epoch in range(100):
                print(f"\rEpoch {epoch}", end="")
                optimizer.zero_grad()
                buffer_tensor = torch.tensor(buffer, dtype=torch.float32, requires_grad=True)
                output = adapter(buffer_tensor)
                loss = loss_function(output[:, 0].float(), buffer_tensor[:, -1].float())
                losses.append(loss.item())
                loss.backward()
                for param in adapter.parameters():
                    print(param.grad)
                optimizer.step()
            buffer = []

        # if ti % 15 == 0:
        #     target = np.random.rand(1) * 6 + 7
        targets.append(target)

        x0_adj = adapter.state_adjuster(torch.tensor(x0, dtype=torch.float32)).detach().numpy()
        a = controller.compute_control(x0_adj, target)
        a_adj = adapter.action_adjuster(torch.tensor(a, dtype=torch.float32)).detach().numpy()
        controls.append(a_adj)
        x = system.response(x0, a_adj, do_update=True)
        signal.append(x)
        buffer.append([*x0, *a_adj, *x, *target])
        x0 = x

    signal = np.asarray(signal)

    fig, ax = plt.subplots(2, 1, sharex=True)

    ax[0].plot(t, signal[:, 0])
    ax[0].plot(t, targets)
    ax[0].invert_yaxis()

    ax[1].plot(t, controls)
    ax[1].invert_yaxis()

    fig.tight_layout()

    fig2, ax2 = plt.subplots(1)
    ax2.plot(losses)

    if not os.path.exists("./tmp"):
        os.makedirs("./tmp")
    fig.savefig("./tmp/plot.png", dpi=300)
    plt.show()

    # # System identification
    # z = c / (2 * np.sqrt(m * k))
    # wn = np.sqrt(k / m) * np.sqrt(1 - z ** 2)


if __name__ == "__main__":
    # Check if CUDA is available
    if torch.cuda.is_available():
        device = torch.device("cuda")

    main()
