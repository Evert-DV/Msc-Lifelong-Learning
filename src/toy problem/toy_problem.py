import os

os.environ["KERAS_BACKEND"] = "torch"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import matplotlib.pyplot as plt
from keras.saving import load_model
from keras import layers, optimizers, initializers
import torch
from torch import optim
from toy_tools import *


def main():
    # seed = np.random.randint(0, 1000)
    seed = 131
    np.random.seed(seed)
    print(f"Seed: {seed}")
    torch.manual_seed(16)

    system = System(5, 10, 3, 5)
    reference_controller = PIDController(350, 107.5, 1257)
    controller = PIDController(350, 107.5, 1257)

    # ilc_model = keras.Sequential([
    #     layers.Input(shape=(1,)),
    #     # layers.Dense(16, activation='relu', kernel_initializer=initializers.Constant(.2), bias_initializer='zeros'),
    #     layers.Dense(2 * 60 * 15, kernel_initializer=initializers.Constant(.5), bias_initializer='zeros'),
    # ])
    ilc_model = load_model("./tmp/best_model.keras")

    ilc_optim = optim.Adam(ilc_model.parameters(), lr=5.e-3)
    ilc_model.compile()

    x0 = [9.9, 0]  # 9.9 was found to be the steady state
    x0_reference = x0
    target = np.array([11., 0.])

    signal = []
    reference_signal = []
    targets = []
    adapted_targets = []
    reference_controls = []
    adapted_controls = []
    buffer = []
    losses = []
    best_loss = torch.inf

    delta_target = np.zeros((15 * 60, 2))
    old_adaptations = ops.array(delta_target)
    # gain = .5
    # previous_error = 0.
    old_loss = ops.zeros((15 * 60 * 2))
    # ilc_gains = []

    i = 0
    dt = 1 / 60
    t = np.arange(0, 600, dt)

    for ti in t:
        # Reset the task every 15 seconds
        if ti % 15 == 0:
            x0 = [9.9, 0]
            x0_reference = x0

        if ti % 15 == 0 and ti != 0:
            i = 0
            print(f"\nt = {ti:.0f}\t\tFitting model...")
            buffer = ops.array(buffer)
            references = buffer[:, -2:]  # the target
            responses = buffer[:, 3:5]  # the (result) state

            # delta_target, gain, previous_error = predictive_ilc(responses, references, previous_error, gain=gain)
            update_ilc = (ti > 15)
            if ti == 30:
                old_adaptations = ops.zeros((15 * 60, 2))
            delta_target, old_adaptations, old_loss = ilc_nn(responses, references, ilc_model, ilc_optim, old_loss,
                                                             old_adaptations, update_ilc)
            losses.append(ops.mean(old_loss ** 2))
            if ops.mean(old_loss ** 2).item() < best_loss and update_ilc:
                best_loss = ops.mean(old_loss ** 2)
                ilc_model.save("./tmp/best_model.keras")
                print("Saved best model")

            buffer = []

        targets.append(target)
        adapted_target = target + ops.convert_to_numpy(delta_target[i])
        control_action = controller.compute_control(x0, adapted_target, dt)
        reference_control = reference_controller.compute_control(x0_reference, target, dt)
        reference_controls.append(reference_control)
        adapted_controls.append(control_action)

        x = system.response(x0, control_action, do_update=False)
        x_reference = system.response(x0_reference, reference_control, do_update=False)

        adapted_targets.append(adapted_target)

        signal.append(x)
        reference_signal.append(x_reference)
        buffer.append([*x0, control_action, *x, *target])

        x0 = x
        x0_reference = x_reference

        i += 1

    signal = np.asarray(signal)
    reference_signal = np.asarray(reference_signal)
    targets = np.asarray(targets)
    adapted_controls = np.asarray(adapted_controls)
    adapted_targets = np.asarray(adapted_targets)

    fig, ax = plt.subplots(3, 1, sharex=True)

    ax[0].plot(t, signal[:, 0], label="Adaptive controller")
    ax[0].plot(t, reference_signal[:, 0], label="Default controller")
    ax[0].plot(t, targets[:, 0], '--', label="Target position")
    ax[0].invert_yaxis()
    ax[0].legend()

    # ax[1].plot(t, adapted_controls, label="Adapted control actions")
    # ax[1].plot(t, reference_controls, label="Reference control actions")
    ax[1].plot(t[::15 * 60][:-1], ops.convert_to_numpy(losses), label="Loss")
    ax[1].legend()

    ax[2].plot(t, adapted_targets[:, 0], '--', label="Adapted target position")
    ax[2].plot(t, targets[:, 0], '--', label="Target position")
    # ax[2].plot(t, ilc_gains, label="ILC gain")
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
