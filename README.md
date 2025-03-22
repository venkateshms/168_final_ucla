# Gene Regulatory Network Reconstruction

This project implements several deep learning approaches for reconstructing Gene Regulatory Networks (GRNs) from interaction data. The framework includes implementations of Graph Neural Networks (GNNs), Convolutional Neural Networks (CNNs), Transformers, and a hybrid Graph+Transformer approach.

## Project Structure

- `preprocess.py`: Script for preprocessing GRN data and splitting it into train/validation/test sets
- `models.py`: Implementation of all model architectures
- `utils.py`: Utility functions for evaluation, metrics tracking, and visualization
- `train.py`: Script for training and evaluating individual models
- `run_experiment.py`: Script for running comparative experiments between different models

## Installation

```bash
pip install -r requirements.txt
```

## Input Data Format

The input data is expected to be a CSV file with the following columns:
- `TF`: Transcription Factor (source node)
- `target`: Target gene (destination node)
- `importance`: Importance/confidence score for the regulatory relationship

## Usage

### Preprocessing

```bash
python preprocess.py --input SCING_GRN.csv --output processed_data
```

### Training a Single Model

```bash
python train.py --model gnn --task link_prediction
```

### Running a Comparative Experiment

To compare all models:

```bash
python run_experiment.py --preprocess --models all
```

To run specific models:

```bash
python run_experiment.py --models gnn,transformer
```

## Model Types

The following models are implemented:

1. **Graph Neural Networks (GNN)**
   - Supports GCN, GAT, GraphSAGE, and GIN architectures
   - Learns node embeddings and predicts edges based on node pairs

2. **Convolutional Neural Networks (CNN)**
   - Uses 1D convolutions to process edge features
   - Can capture local patterns in regulatory relationships

3. **Transformers**
   - Applies self-attention to learn complex relationships between genes
   - Can capture long-range dependencies in the regulatory network

4. **Graph + Transformer Hybrid**
   - Combines GNN's structural learning with Transformer's sequence modeling
   - Uses GNN to learn initial embeddings and Transformer to refine them

## Tasks

The framework supports two main tasks:

1. **Link Prediction**: Predicting whether a regulatory relationship exists between a TF and a target
2. **Regression**: Predicting the importance/confidence score of a regulatory relationship

## Evaluation Metrics

For link prediction:
- ROC AUC
- Precision-Recall AUC
- F1 Score
- Accuracy
- Precision
- Recall

For regression:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score

## Computational Tracking

The framework tracks computational complexity:
- Parameter counts for each model
- FLOPs for inference
- Inference time

## Results

After running experiments, results and visualizations are saved to:
- `logs/results/`: JSON files with detailed metrics
- `logs/plots/`: Comparison plots for different models

