import os

os.environ["KERAS_BACKEND"] = "torch"
import matplotlib.pyplot as plt
from keras.api.saving import load_model
from torch.utils.data import DataLoader
from src.toy_problem.toy_tools import *


def main():
    pretrain = True
    incremental_updates = True

    seed = np.random.randint(0, 1000)
    # seed = 63
    print(f"Seed: {seed}")
    np.random.seed(seed)
    torch.manual_seed(seed)
    if pretrain:
        torch.manual_seed(16)

    system = System(5, 10, 3, 5)
    controller = PIDController(350, 107.5, 1257)
    reference_controller = PIDController(350, 107.5, 1257)

    prediction_window = 10
    adapter = TargetAdapter(state_size=2, target_size=1)

    model_location = '../../tmp/target_adapter.weights.h5'
    if not pretrain:
        adapter.load_weights(model_location)
        # adapter.regularizer.add(RMSERegularizer(weight=.75))

    optimizer = keras.optimizers.Adam(learning_rate=5.e-3)
    loss_fn = keras.losses.MeanSquaredError()
    adapter.compile(optimizer=optimizer, loss=loss_fn)

    if pretrain:
        pretrain_data = np.load("../../tmp/sim data/m5k10c3_seed358.npy")
        train_set, val_set = prep_data(pretrain_data, prediction_window, state_size=2, target_size=1, val_split=0.2)
        train_dataloader = DataLoader(train_set, batch_size=256, shuffle=True)
        val_dataloader = DataLoader(val_set, batch_size=256, shuffle=False)

        callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss',
                                                   mode='min',
                                                   min_delta=1e-4,
                                                   patience=5,
                                                   restore_best_weights=True,
                                                   verbose=1),
                     ]
        adapter.fit(train_dataloader,
                    epochs=100,
                    callbacks=callbacks,
                    validation_data=val_dataloader,
                    )

        adapter.save_weights(model_location)

    dt = 1 / 50
    x0 = [9.9, 0]  # 9.9 was found to be the steady state
    signal = []
    reference_signal = []
    targets = []
    adapted_targets = []
    reference_controls = []
    adapted_controls = []
    predicted_targets = []
    t = np.arange(0, 300, dt)
    target = np.array([11., 0.])
    buffer = []
    x0_reference = [9.9, 0]

    if not pretrain:
        for ti in t:
            if ti % 15 == 0 and ti != 0:
                adapter.optimizer.lr = 1.e-3
                buffer = ops.array(buffer)
                features, labels = prep_data(buffer, prediction_window, state_size=2, target_size=1)
                ref_prediction = adapter.predict(features, verbose=0)
                predicted_targets += ref_prediction[:, 0].ravel().tolist()
                predicted_targets += prediction_window * [float('nan')]

                if incremental_updates:
                    print("\nFitting model...")
                    train_dataset, val_dataset = random_split(TensorDataset(features, labels),
                                                              [int(0.8 * len(features)),
                                                               len(features) - int(0.8 * len(features))])
                    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
                    val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=False)

                    callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss',
                                                               mode='min',
                                                               min_delta=1e-4,
                                                               patience=5,
                                                               restore_best_weights=True,
                                                               verbose=1),
                                 EpochLogger()
                                 ]
                    adapter.fit(train_dataloader,
                                epochs=100,
                                callbacks=callbacks,
                                validation_data=val_dataloader,
                                verbose=0,
                                )

                buffer = []

            if ti % 5 == 0:
                # target = [7, 0.] if ti % 2 == 0 else [13, 0.]
                target = [np.random.uniform(5, 15), 0.]
                # target = [np.random.rand() * 6 + 7, 0.]

            targets.append(target)

            predicted_target = adapter.predict(ops.array([*x0, *target])[None], verbose=0)[0]
            adapted_targets.append(predicted_target)
            control_action = controller.compute_control(x0, target + predicted_target, dt)
            adapted_controls.append(control_action)
            x = system.response(x0, control_action, do_update=True)
            signal.append(x)
            x0 = x

            # Reference control loop
            reference_control = reference_controller.compute_control(x0_reference, target, dt)
            reference_controls.append(reference_control)
            x_reference = system.response(x0_reference, reference_control, do_update=True)
            reference_signal.append(x_reference)
            x0_reference = x_reference

            buffer.append([*x0, control_action, *x, *(target + predicted_target)])

        # adapter.save(model_location)

        signal = np.asarray(signal)
        reference_signal = np.asarray(reference_signal)
        targets = np.asarray(targets)
        adapted_targets = np.asarray(adapted_targets)
        predicted_targets = np.asarray(predicted_targets).ravel()
        adapted_controls = np.asarray(adapted_controls)

        fig, ax = plt.subplots(3, 1, sharex=True)

        ax[0].plot(t, reference_signal[:, 0], color='lightgrey', label="Reference controller")
        ax[0].plot(t, targets[:, 0], '--', color='tab:gray', label="Target position")
        ax[0].plot(t, signal[:, 0], color='tab:blue', label="Adaptive controller")
        ax[0].set_xlim(10, 80)
        ax[0].invert_yaxis()
        ax[0].legend(fontsize=8, loc='upper left')

        ax[1].plot(t, reference_controls, color='lightgrey', label="Reference control actions")
        ax[1].plot(t, adapted_controls, color='tab:blue', label="Adapted control actions")
        ax[1].legend(fontsize=8, loc='upper left')

        ax[2].plot(t, targets[:, 0], '--', color='tab:gray', label="Target position")
        ax[2].plot(t, adapted_targets[:, 0], color='tab:blue', label="Adapted targets")
        ax[2].plot(t[:-15 * 50], predicted_targets, ':', color='tab:orange', label="Predicted targets")
        ax[2].invert_yaxis()
        ax[2].legend(fontsize=8, loc='upper left')

        fig.tight_layout()
        if not os.path.exists("tmp"):
            os.makedirs("tmp")
        # fig.savefig("tmp/plot.png", dpi=300)
        plt.show()


if __name__ == "__main__":
    print("Using backend " + keras.backend.backend())
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    if torch.cuda.is_available():
        print("Using CUDA")
        with torch.cuda.device(0):
            main()
    else:
        main()
