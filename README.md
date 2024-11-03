# Drug Design and Pipeline Development for Proteinopathies

This pipeline is designed for drug discovery targeting protein aggregation inhibitors, specifically aimed at neurodegenerative diseases such as Alzheimer’s and Parkinson’s. The pipeline integrates molecular simulations and machine learning methods to evaluate potential drug candidates based on their effects on protein structures.

## Overview

Protein aggregation plays a key role in the development of neurodegenerative diseases. By predicting the impact of drug-like compounds on these proteins, this pipeline aims to assist researchers in identifying promising therapeutic candidates. The pipeline is modular, enabling clear separation of tasks such as data preparation, embedding generation, and model evaluation.

## Pipeline Structure

The pipeline includes five primary Python scripts and supporting data files:
- **data_preparation.py**: Prepares and scales peptide data.
- **embedding_generator.py**: Generates embeddings for peptide sequences using ProtTrans.
- **main.py**: Orchestrates the entire workflow.
- **model_evaluation.py**: Evaluates and optimizes machine learning models.
- **predictor.py**: Predicts Rg values for new peptide sequences.

### Data Files
- **peptides_rg.csv**: Contains peptide data used for model training and evaluation.
- **new_peptides.csv**: Place new peptide sequences here for prediction.

## Key Objectives

1. Develop structural and dynamic descriptors to assess potential inhibitors of protein aggregation.
2. Implement a pipeline combining molecular simulations and machine learning to evaluate these drug candidates effectively.

## Methods

### 1. Molecular Simulations
- **GROMACS**: Used for molecular dynamics simulations, assessing protein structures and dynamics with coarse-grained and all-atom force fields.

### 2. Structural Descriptors
- **Radius of Gyration Analysis** and other techniques are used to identify critical structural changes(not applied in the pipeline yet).
- Custom analysis tools in Python integrate these analyses into the pipeline.

### 3. Evaluation of Drug Candidates
- **Aggregation Analysis**: Potential drugs are analyzed based on their impact on protein aggregation through molecular dynamics.

## Workflow

The pipeline is organized as follows:

1. **Embedding Generation**:
   - `embedding_generator.py` processes each peptide sequence using ProtTrans, generating embeddings that capture structural properties.

2. **Data Preparation**:
   - `data_preparation.py` merges peptide data with embeddings, scales features, and saves the dataset as `MLdata.csv` for model training.

3. **Model Evaluation**:
   - `model_evaluation.py` evaluates multiple ML models (Linear Regression, Random Forest, etc.) and tunes them for optimal performance, saving the best model and scaler for future predictions.

4. **Prediction of New Peptides**:
   - `predictor.py` loads the best model to predict Rg values for peptides in `new_peptides.csv`, saving the predictions in `predicted_peptides.csv`.

## Running the Pipeline

To run the pipeline, follow these steps:

1. Place new peptide sequences in `new_peptides.csv`.
2. Run `main.py` to execute the pipeline:
   ```bash
   python3 main.py
