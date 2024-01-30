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
    seed = np.random.randint(0, 1000)
    # seed = 42
    np.random.seed(seed)
    print(f"Seed: {seed}")
    torch.manual_seed(42)

    system = System(5, 10, 3, 5)
    reference_controller = PIDController(350, 107.5, 1257)
    controller = PIDController(350, 107.5, 1257)

    classic_ilc = True
    pred_ilc = False
    nn_ilc = False
    nn_target = False

    if nn_ilc or nn_target:
        model_location = "./tmp/best_model.keras" if nn_ilc else "./tmp/target_adapter.keras"
        # model = keras.Sequential([
        #     layers.Input(shape=(1,)),
        #     layers.Dense(64, activation='sigmoid', kernel_initializer=initializers.RandomUniform(-0.1, 0.1),
        #                  bias_initializer=initializers.RandomUniform(-.1, .1)),
        #     layers.Dropout(0.1),
        #     layers.Dense(64, activation='relu', kernel_initializer=initializers.RandomUniform(-0.1, 0.1),
        #                  bias_initializer=initializers.RandomUniform(-.1, .1)),
        #     layers.Dropout(0.1),
        #     layers.Dense(2 * 60 * 15, kernel_initializer=initializers.RandomUniform(-0.1, 0.1),
        #                  bias_initializer=initializers.RandomUniform(-.1, .1)),
        # ])
        model = load_model(model_location)
        optimizer = optim.Adam(model.parameters(), lr=1.e-3)
        model.compile()

        losses = []
        best_loss = torch.inf

        delta_target = np.zeros((15 * 60, 2))
        old_delta = ops.array(delta_target).ravel()
        old_errors = ops.zeros((15 * 60 * 2))

        if nn_ilc:
            persistent_adaptations = np.load('./tmp/adaptations.npy')
            # persistent_adaptations = np.zeros((15 * 60, 2))

    elif classic_ilc:
        gain = .5
        previous_error = 0.
        ilc_gains = []
        delta_target = np.zeros((15 * 60, 2))

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

    dt = 1 / 60
    t = np.arange(0, 600, dt)
    i = 0

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

            if classic_ilc:
                delta, gain, previous_error = simple_ilc(responses, references, previous_error, gain=gain)
                delta_target += delta

            elif pred_ilc:
                delta_target, gain, previous_error = predictive_ilc(responses, references, previous_error, gain=gain)

            elif nn_target or nn_ilc:
                update_ilc = (ti > 15)
                if ti == 30:
                    old_delta = ops.zeros((15 * 60 * 2))

                if nn_ilc:
                    delta_target, old_delta, old_errors = ilc_nn(responses, references, model, optimizer, old_errors,
                                                                 old_delta, update_ilc)
                    persistent_adaptations += ops.convert_to_numpy(delta_target)
                    delta_target = persistent_adaptations

                elif nn_target:
                    old_delta, old_errors = train_iterative(model, optimizer, ilc_loss_fn, references, responses,
                                                            old_delta, old_errors)

                    delta_target = model.predict(ops.ones(1)[None], verbose=0)[0]
                    delta_target = ops.convert_to_numpy(delta_target).reshape(-1, 2)

                losses.append(ilc_loss_fn(old_errors))
                if ilc_loss_fn(old_errors).item() < best_loss and update_ilc:
                    best_loss = ilc_loss_fn(old_errors)
                    model.save(model_location)
                    # np.save("./tmp/adaptations.npy", persistent_adaptations)
                    print("Saved best model")

            buffer = []

        targets.append(target)
        adapted_target = target + delta_target[i]
        adapted_targets.append(adapted_target)
        control_action = controller.compute_control(x0, adapted_target, dt)
        adapted_controls.append(control_action)
        x = system.response(x0, control_action, do_update=False)
        signal.append(x)
        x0 = x

        # reference loop
        reference_control = reference_controller.compute_control(x0_reference, target, dt)
        reference_controls.append(reference_control)
        x_reference = system.response(x0_reference, reference_control, do_update=False)
        reference_signal.append(x_reference)
        x0_reference = x_reference

        buffer.append([*x0, control_action, *x, *target])
        i += 1

    signal = np.asarray(signal)
    reference_signal = np.asarray(reference_signal)
    targets = np.asarray(targets)
    adapted_controls = np.asarray(adapted_controls)
    adapted_targets = np.asarray(adapted_targets)

    fig, ax = plt.subplots(3, 1, sharex=True)

    ax[0].plot(t, reference_signal[:, 0], label="Default controller")
    ax[0].plot(t, signal[:, 0], label="Adaptive controller")
    ax[0].plot(t, targets[:, 0], '--', label="Target position")
    ax[0].invert_yaxis()
    ax[0].legend()

    ax[1].plot(t, adapted_controls, label="Adapted control actions")
    ax[1].plot(t, reference_controls, label="Reference control actions")
    ax[1].legend()

    ax[2].plot(t, targets[:, 0], '--', label="Target position")
    ax[2].plot(t, adapted_targets[:, 0], '--', label="Adapted target position")
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
