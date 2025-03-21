"""
This file stores all customizable configuration parameters for your federated learning
experiments. Adjust these values as needed before running the Server and Client scripts.
"""

config = {
    # =======================================================
    # Data Configuration
    # =======================================================
    "criteria": "state_class",  # Options: "random" for IID clients, "zone_expl" for Bioclimatic Regions, "state_class" for Geographical Regions
    "spectra_type": "MIR",      # Options: "VNIR", "MIR"
    "soil_properties": [
        "SOC",
        "clay",
        "sand",
        "PH",
        "CEC_cmol.kg",
        "TN_.wt"
    ],
    # Client Data paths
    "client_data": "data/test_dataset.csv",


    # Server Data paths
    "path_X_VNIR": "data/global_XMIR.csv",
    "path_X_MIR":  "data/global_XMIR.csv",
    "path_y":      "data/global_y_dataset.csv",

    # Number of clients
    # (4 for IID or Regions, 7 for BCL, but can be adjusted as desired)
    "num_clients": 4,

    # =======================================================
    # Federated Learning / Flower Server Configuration
    # =======================================================
    "strategy": "FedAvg",        # Options: "FedAvg", "WgtAvg"
    "server_address": "localhost:8000",
    "num_rounds": 100 ,           # Total federated training rounds

    # =======================================================
    # Model Configuration
    # =======================================================
    "model_params": {
        # CNN architecture details
        "filters1": 32,
        "filters2": 64,
        "kernel_size1": 7,
        "kernel_size2": 7,

        # Training hyperparameters
        "optimizer": "adam",      # e.g., "adam", "sgd"
        "learning_rate": 0.0001,
        "momentum": 0.1,          # if using SGD with momentum
        "batch_size": 64,
        "epochs": 100,            # local epochs per round
        "early_stop_patience": 50 # patience for EarlyStopping callback
    },

    # =======================================================
    # Miscellaneous
    # =======================================================
    "seed": 42,                  # Random seed for reproducibility
    "input_shape": 353,          # Input dimension (e.g., 215 for VNIR)
    "output_shape": 6,           # Number of soil properties to predict
    "val_ratio": 0.3,            # Validation Ratio for local KS split

    # Directory paths for saving model weights, metrics, etc.
    "save_weights_dir": "weights_region",
    "metrics_dir": "metrics_region"
}
