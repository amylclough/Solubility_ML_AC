# Solubility Machine Learning Training and Predictor Tool Based Of Final Model

Overview.

** Author ** : Amy Clough, University of Southampton


## One Time Envioment Setup and Installation Instructions

1. Ensure you have Python installed on your system.

2. Open your terminal or command prompt and navigate to the folder containing these files.

3. Create a virtual environment 
```bash
python3 -m venv solubility_ml
```

4. Activate the virtual environment
- On Linux/macOS
```bash
source solubility_ml/bin/activate
```
- on windows
```bash
solubility_ml\Scripts\activate
```

5. Install the required dependencies by running the following command:
```bash
pip install -r requirements.txt
```

## Directory Structure:

## Run Predictions:
1. Open your terminal in the predictions folder

2. Run the script using Python:
```bash
python Prediction.py
```

3. You will be promoted to enter a SMILES string

4. You can optionally specify how many features you would like displayed in the SHAP waterfall (it defaults to 10)

5. You can optionally type in a specific feature name (e.g MolWt, MACCS_12, Si) to see exactly how much it impacted the prediction

6. A local HTML file will be produced when the prediction is complete, this file can be opened in any web browser to view the visual results. 

## Data Availability

PSDI Database information.
To link to known drug names from SMILES, access the drugbank structure_links.csv file and save in this directory. Due to licencing this file cannot be publicly distributed here. The prediction script still runs without this file, however will not give you the name of the compond name.
