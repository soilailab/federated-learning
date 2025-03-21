# Federated Learning with Flower Framework for Soil Spectroscopy

This repository implements a Federated Learning (FL) system using the Flower (FLWR) framework to train a machine learning model on partitioned datasets related to soil spectroscopy. The system is designed to perform training on a distributed network of clients, where each client holds its local dataset and performs training without sharing the raw data, ensuring privacy and security.

## Table of Contents
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [How Federated Learning Works](#how-federated-learning-works)
- [Data Partitioning](#data-partitioning)
- [Aggregators](#aggregators)

--- 

## Installation

To run this project, you'll need the following dependencies installed. It is recommended to use a virtual environment.

### 1. Clone the repository:

```bash
git clone https://github.com/soilailab/soil-spectroscopy-fl.git
cd soil-spectroscopy-fl
```

### 2. Install required Python packages:

```bash
pip install -r requirements.txt
```

## Project Structure
The project consists of several files and modules:

- `client.py`: Defines the client-side logic for federated learning, including model training and evaluation.
- `server.py`: Defines the server-side logic, where model parameters are aggregated and global evaluation is performed.
- `model.py`: Contains the CNN model architecture and the compilation logic.
- `dataset.py`: Includes functions for loading, partitioning, and splitting the dataset among clients.
- `custom_aggregators.py`: Implements custom aggregation functions such as FedAvg and WgtAvg for model parameter aggregation.
- `config.py`: Configuration file to define hyperparameters, paths, and other settings.
- `requirements.txt:` A file listing all dependencies for the project.

## Usage

### 1. Modify configuration file
Before running the scripts, modify the `config.py` file to match your setup:
- Define the paths to your dataset.
- Set the number of clients and partitioning criteria.
- Adjust model parameters like filters, kernel sizes, etc.

### 2. Running the Flower server
To start the Flower server, run the following command:
```bash
python server.py
```

### 3. Running the Flower client
To run the client, use:
```bash
python client.py --cid <client_id>
```
where `<client_id>` is an integer representing the client ID (e.g. `0`, `1`, `2` etc).


### 4. Monitoring and Evaluation:
The server periodically evaluates the global model on a test dataset, and the metrics are saved to `metrics_dir`. Each client trains its local model and sends updates to the server for aggregation.


## How Federated Learning Works
<b>1. Client Initialization</b>: Each client loads its local dataset and trains a local model on it. Clients communicate with the server but never share their raw data.

<b> 2. Model Aggregation</b>: Once the clients finish training their models, they send their model updates (parameters) to the server.

<b> 3. Global Model Update</b>: The server aggregates the model updates using either a weighted average (WgtAvg) or a simple average (FedAvg).

<b> 4. Evaluation</b>: The server evaluates the global model on a test set and periodically logs the evaluation metrics.


## Data Partitioning

The dataset is partitioned among the clients based on different strategies:

- <b>IID</b>: Data is randomly shuffled and split equally among all clients.
- <b>BCL</b>: Data is partitioned by the zone_expl column.
- <b>Regions</b>: Data is partitioned by the state_class column.

Partitioning is handled in `dataset.py`, and the client receives its specific subset of the data based on the partitioning strategy.


## Aggregators
The model parameters are aggregated on the server using two strategies:

- <b>FedAvg (Federated Averaging)</b>: Computes a simple unweighted average of model updates across all clients.

- <b>WgtAvg (Weighted Averaging)</b>: Computes a weighted average of model updates, where the weight is determined by the number of examples used by each client.

These aggregation functions are implemented in `custom_aggregators.py`.


