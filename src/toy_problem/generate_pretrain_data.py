import os

os.environ["KERAS_BACKEND"] = "torch"
import numpy as np
import matplotlib.pyplot as plt
from src.toy_problem.toy_tools import PIDController, System


def main():
    seed = np.random.randint(0, 1000)

    print(f"Seed: {seed}")
    np.random.seed(seed)

    x0 = [9.9, 0]  # 9.9 was found to be the steady state
    target = np.array([11., 0.])
    signal = []
    targets = []
    controls = []
    data = []

    controller = PIDController(700, 50, 1000)
    m = 5
    k = 20
    c = .5
    # default values: m = 5, k = 10, c = 3, l0 = 5
    system = System(m, k, c, 5)
    system_updates = False

    dt = 1 / 50
    t = np.arange(0, 300, dt)

    for ti in t:
        if ti % 5 == 0:
            target = [np.random.uniform(5, 15), 0.]

        targets.append(target)
        control_action = controller.compute_control(x0, target, dt)
        x = system.response(x0, control_action, do_update=system_updates)
        signal.append(x)
        data.append([*x0, control_action, *x, *target])

        x0 = x

    data = np.asarray(data)
    np.save(f"../../tmp/sim data/m{m}k{k}c{c}_{'w-update_' if system_updates else ''}seed{seed}.npy", data)
    print(f"Data saved in tmp/sim data/m{m}k{k}c{c}_{'w-update_' if system_updates else ''}seed{seed}.npy")

    signal = np.asarray(signal)
    targets = np.asarray(targets)

    plt.plot(t, signal[:, 0], label="signal")
    plt.plot(t, targets[:, 0], '--', label="target")
    plt.show()


if __name__ == '__main__':
    main()
