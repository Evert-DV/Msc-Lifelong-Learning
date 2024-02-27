from kb_tools import *
from toy_tools import *


def main():
    # Load adapter model
    #   - optimizer
    #   - loss function
    #   - compile
    # Load VAE model
    #   - optimizer
    #   - loss function
    #   - compile
    # Define system
    # Define controller
    # Define reference controller
    # Setup KB
    # Setup simulation loop

    # Simulation loop
    #   Pre-60 seconds
    #       Simulation step
    #       Gather transitions
    #       Get VAE embeddings
    #       Get KB posteriors
    #   Select KB entry
    #   Post-60 seconds
    #       Simulation step
    #       Gather transitions
    #       Get VAE embeddings
    #       Get JS divergence running to prior
    #       Check environment shift*
    #   Update step (every n seconds)
    #       Update running distribution
    #       Check KB for better entry
    #       Update prior distribution
    #       Update target adapter
    pass


if __name__ == '__main__':
    print("Using backend " + keras.backend.backend())
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

    model_location = 'tmp/varautoencoder.keras'
    seed = np.random.randint(0, 1000)
    # seed = 267
    print(f"Seed: {seed}")
    keras.utils.set_random_seed(seed)

    if torch.cuda.is_available():
        print("Using CUDA")
        with torch.cuda.device(0):
            main()
    else:
        main()
