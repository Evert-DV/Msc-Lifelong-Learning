import os

os.environ["KERAS_BACKEND"] = "torch"
import torch
import keras
import numpy as np
from keras.src.saving import load_model
from torch.utils.data import DataLoader
from src.toy_problem.toy_tools import TargetAdapter, prep_data


def main():
    pretrain = False

    # seed = np.random.randint(0, 1000)
    seed = 63
    print(f"Seed: {seed}")
    np.random.seed(seed)
    torch.manual_seed(seed)
    if pretrain:
        torch.manual_seed(16)

    prediction_window = 10

    model_location = './Models/'
    model_name = 'adapter_weights.weights.h5'

    adapter = TargetAdapter(state_size=1)

    if not pretrain:
        adapter.load_weights(f"{model_location}{model_name}")
        adapter.summary()
        # adapter = torch.load(model_name)
        print("Model loaded")
        # adapter.regularizer.add(RMSERegularizer(weight=.75))

    optimizer = keras.optimizers.Adam(learning_rate=5e-4)
    loss_fn = keras.losses.MeanSquaredError()
    adapter.compile(optimizer=optimizer, loss=loss_fn)

    pretrain_data = np.load("./Dynamics data/75-75 perforation/b_auto_save_3.npy")
    train_set, val_set = prep_data(pretrain_data, prediction_window, state_size=1, val_split=0.2)
    train_dataloader = DataLoader(train_set, batch_size=256, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=256, shuffle=False)

    callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss',
                                               mode='min',
                                               min_delta=1e-4,
                                               patience=10,
                                               restore_best_weights=True,
                                               verbose=1),
                 ]
    adapter.fit(train_dataloader,
                epochs=1000,
                callbacks=callbacks,
                validation_data=val_dataloader,
                )

    adapter.save_weights(f"{model_location}{model_name}")
    # torch.save(adapter, model_name)
    adapter.summary()
    print("Model saved")


if __name__ == "__main__":
    print("Using backend " + keras.backend.backend())
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    if torch.cuda.is_available():
        print("Using CUDA")
        with torch.cuda.device(0):
            main()
    else:
        main()
