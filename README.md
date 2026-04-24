# Machine Learning Models for Aqueous Solubility Prediction

**Author**: Amy Clough, University of Southampton

## Abstract

Solubility prediction remains a complex scientific challenge and therefore an active research area, with applications in many fields. This project focuses on predicting aqueous solubility through machine learning models. Solubility prediction is affected by many different factors including experimental data uncertainty and the availability of data. Many different models using different algorithms, features and training data have been produced but comparing these models proves a challenge. This project investigates the importance of using highly curated data to train a model, the use of experimental data (melting point and enthalpy of fusion) and type of algorithm. A machine learning model for predicting solubility using a range of features including 2D physicochemical descriptors, atom counts and molecular fingerprints is then presented. The performance of the model is universally tested alongside subsets of known approved drug molecules, molecules within key pharmaceutical ranges and an external pharmaceutical validation set. The final model produced had comparable results when compared to open-access available models, achieving accuracy generally within experimental error. When compared to simple methods for estimating solubility directly from the structure, Estimated SOLubility (ESOL) and the widely used General Solubility Equation (GSE), the project model exhibited improved predictions. This model gives a greater degree of transparency in the model’s decisions than other models in the field making it well suited for drug discovery. A final prediction script is available for download and use which outputs a predicted value, an applicability domain as well as global and local analysis into the model’s decisions.

## Introduction

This repository contains a comprehensive machine learning pipeline for predicting aqueous solubility (LogS) of chemical compounds, primarily focused on drug-like molecules. The project uses cheminformatics descriptors, molecular fingerprints, and machine learning techniques to develop predictive models with interpretability features. The work builds upon multiple datasets and iterative model development to achieve predictions suitable for pharmaceutical applications.

The project objectives are as followed:
- To investigate current ML models and the importance of training a model on highly curated and cleaned data.
- To select appropriate features and preprocess data in order to generate a solubility prediction ML method.
- To build a ML model for predicting solubility with a downloadable prediction script and evaluate its performance against experimental uncertainty and uncertainty of published models.
- To investigate the model performance for known pharmaceutical molecules and molecules which are within key related pharmaceutical ranges

## Datasets

The models are trained and validated using several publicly available and proprietary datasets:

- **PChProp**: A Solubility dataset by the PDSI 
- **AqSolDB**: A curated database of aqueous solubility measurements for drug-like molecules.
- **AqSolDB(c)**: A corrected version of AqSolDB with improved data quality from Llompart et al (2024).
- **DrugBank**: A comprehensive database of drug information, used for validation and linking SMILES to known drug names.

Data preprocessing involves cleaning, feature engineering, and removal of outliers, as detailed in the `Data Cleaning/` directory.

## Methods
For the full methods section please see the full dissertation, access can be provided upon request. 
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

## Conclusion

Within this project, a series of investigative models were produced leading to decisions informing the final model design. Whilst the use of experimental features was considered, the decision was made to avoid using them as there was a large reduction in available
training data limited model improvement. This decision also meant that the model was still feasible for new compounds without experimental data for example use within drug discovery pipelines. The importance of data curation was investigated and seen to be significant in model performance, which meant a robust data cleaning pipeline was designed and applied to the most recent version of PChProp prior to model training.
A series of XGBoost models were compared testing the inclusion of different features and the affect on both the performance and transparency of the final model. The final model was made from the following categories: 2D molecular descriptors, MACCS fingerprints and custom atom and bond counts with the decision being made to not include AP and MF.
The output of the project is a downloadable and easy to use prediction script, available here (Prediction_Script folder). This script allows the user to enter a SMILES string of a proposed molecule and receive an output HTML. The output HTML contains the following features: Predicted LogS, the name of the molecule if it is a FDA approved drug (in DrugBank database), a 2D drawing of a molecule, a confidence score (similarity to training set) as well as performance metrics for this region of similarity, a SHAP waterfall plot to show how the prediction was made of customisable length and specific requested feature summaries. Since only calculated features are used to train the model it can be used for potential
molecules making it a viable tool for use within drug discovery pipelines.
The ML model predicted the LogS of molecules with a MAE of 0.575 for the whole test set, 0.625 for the subset of approved drugs within this and 0.648 for an external validation set made of approved drugs. 72.9% of predictions for the whole test set were within experimental error and this was also reflected in the subset and external validation set with values of 68.7% and 69.5% respectively. This performance shows that for each
set the average error was under experimental data, showing good performance. The downloadable prediction script makes this a viable tool for predicting solubility, allowing for potential use in pharmaceutical research in particular.

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
- `Bonds/`: Model variant exploring bond-based and ring size features.
- `Data Cleaning/`: Scripts for data preprocessing and exploration.
- `DrugBank/`: Analysis of Lipinski violations and drug properties.
- `ESOL/`: Evaluation of ESOL linear regresion model and fitting to new data.
- `Final Model/`: Main trained model, prediction scripts, and evaluation reports.
- `Initial Models/`: Jupyter notebooks for exploratory data analysis and initial modeling.
- `MF Test/`: Model comparison with different feature sets.
- `PChProp V1/`: Jupyter notebook for initial data exploration of early stage PChProp.
- `Prediction_Script/`: Standalone prediction tool based off of Final Model.
- `requirements.txt`: Python dependencies.
- `README.md`: This file.

Note: to view the HTML report files, download them and open in your browser as GitHub does not natively support a HTML view.

## Data Availability

- Core datasets (AqSolDB, AqSolDBc, thermodyanmic) are publicly available and referenced in the code. Downloadable from the orginal papers:
  M. C. Sorkun, A. Khetan and S. Er, Sci Data, 2019, 6, 143
  P. Llompart, C. Minoletti, S. Baybekov et al., Scientific Data, 2024, 11, 303
  E. Al Ibrahim, N. Morgan, S. Müller, S. Motati and W. Green, Accurately predicting solubility curves via a thermodynamic cycle, machine learning,    and solvent ensembles, ChemRxiv Preprint, Preprint. Not peer-reviewed, 2025 (since published but data was taken at time from this version)
- DrugBank structure links (`structure_links.csv`) can be obtained from DrugBank but are not distributed here due to licensing. The prediction tool functions without it but cannot provide compound names. https://go.drugbank.com/releases/latest
- PChProp data: For information and access to either version of the PChProp data contact Dr. Matthew Partridge or Prof. Jeremy Frey data collection.
  Further information can be found: M. Partridge and J. Frey, Physical Chemistry Properties Data Collection, Zenodo, 2025
