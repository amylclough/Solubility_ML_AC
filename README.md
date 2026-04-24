# Machine Learning Models for Aqueous Solubility Prediction

**Author**: Amy Clough, University of Southampton

## Abstract

[to be copied in]

## Introduction

This repository contains a comprehensive machine learning pipeline for predicting aqueous solubility (LogS) of chemical compounds, primarily focused on drug-like molecules. The project uses cheminformatics descriptors, molecular fingerprints, and machine learning techniques to develop predictive models with interpretability features. The work builds upon multiple datasets and iterative model development to achieve predictions suitable for pharmaceutical applications.

The primary research aim addressed is: [to be copied in]

## Datasets

The models are trained and validated using several publicly available and proprietary datasets:

- **PSDI Solubility Dataset**: Additional solubility data from the Physical Sciences Data-science Service (PSDI) at https://www.psdi.ac.uk/
- **AqSolDB**: A curated database of aqueous solubility measurements for drug-like molecules.
- **AqSolDB(c)**: A corrected version of AqSolDB with improved data quality.
- **ESOL**: Estimated SOLubility dataset, providing computed solubility values for diverse organic compounds.
- **DrugBank**: A comprehensive database of drug information, used for validation and linking SMILES to known drug names.

Data preprocessing involves cleaning, feature engineering, and removal of outliers, as detailed in the `Data Cleaning/` directory.

## Methods

### Feature Engineering
Molecular features are generated using RDKit, including:
- Standard descriptors (e.g., molecular weight, logP, number of hydrogen donors/acceptors).
- Atom counts for specific elements (C, H, N, O, S, Cl, F, P, I, Br, etc.).
- MACCS (Molecular ACCess System) keys for structural fingerprints.

### Model Development
- **Algorithm**: XGBoost Regressor, optimized using Optuna for hyperparameter tuning (150 trials).
- **Preprocessing**: Standard scaling of features.
- **Evaluation**: K-fold cross-validation, with metrics including R², RMSE, MAE, and % within 0.7 and 1 log units.
- **Explainability**: SHAP (SHapley Additive exPlanations) for feature importance and waterfall plots.

The final model is trained on a cleaned dataset, with feature selection and outlier removal applied. Various model variants are explored in subdirectories (e.g., AC Test, AP Test, Bonds, MF Test) to compare different feature sets and architectures.

### Validation and Testing
- Internal validation using train-test splits.
- External validation on held-out datasets.
- Applicability domain assessment.
- Similarity-based tests for domain of applicability.

## Final Model Results

[Results for the final model to be copied in]

## Usage

### Prediction Tool
The repository includes a standalone prediction script that allows users to input SMILES strings and obtain solubility predictions with explainability reports.

1. Navigate to the `Prediction_Script/` directory.
2. Run the prediction script:
   ```bash
   python Prediction.py
   ```
3. Enter a SMILES string when prompted.
4. Optionally specify the number of top features to display in the SHAP explanation (default: 10).
5. Optionally query specific feature impacts.
6. View the generated HTML report in a web browser for visual results, including confidence intervals and feature contributions.

### Training and Evaluation
To retrain or evaluate models:
- Use scripts in `Final Model/Model_Final.py` for the main model.
- Explore notebooks in `Initial Models/` for exploratory analysis.
- Run evaluation scripts in subdirectories for specific model variants.

## Installation

1. Ensure Python 3.x is installed on your system.

2. Clone or download this repository.

3. Create a virtual environment:
   ```bash
   python3 -m venv solubility_ml
   ```

4. Activate the virtual environment:
   - On macOS/Linux:
     ```bash
     source solubility_ml/bin/activate
     ```
   - On Windows:
     ```bash
     solubility_ml\Scripts\activate
     ```

5. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

Required packages include: pandas, numpy, scikit-learn, xgboost, rdkit, shap, optuna, matplotlib, seaborn, joblib.

## Directory Structure

- `AC Test/`: Model variant with atom count features.
- `AP Test/`: Model variant with aromatic proportion.
- `Bonds/`: Model exploring bond-based features.
- `Data Cleaning/`: Scripts for data preprocessing and exploration.
- `DrugBank/`: Analysis of Lipinski violations and drug properties.
- `ESOL/`: Evaluation on ESOL dataset.
- `Final Model/`: Main trained model, prediction scripts, and evaluation reports.
- `Initial Models/`: Jupyter notebooks for exploratory data analysis and initial modeling.
- `MF Test/`: Model comparison with different feature sets.
- `PChProp V1/`: Initial data exploration.
- `Prediction_Script/`: Standalone prediction tool based off of Final Model.
- `requirements.txt`: Python dependencies.
- `README.md`: This file.

Note: to view the HTML report files, download them and open in your browser as GitHub does not natively support a HTML view.

## Data Availability

- Core datasets (AqSolDB, ESOL) are publicly available and referenced in the code.
- DrugBank structure links (`structure_links.csv`) can be obtained from DrugBank but are not distributed here due to licensing. The prediction tool functions without it but cannot provide compound names.
- PSDI data: https://www.psdi.ac.uk/ - [to be copied in] - access
