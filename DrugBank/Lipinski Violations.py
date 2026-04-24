"""Importing Required Libraries"""
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
import os

# --- 1. Setup and Configuration ---
input_file = 'path/to/DrugBankStructurelinks' 
output_csv = 'lipinski_summary.csv'

if not os.path.exists(input_file):
    print(f"Error: {input_file} not found in the current directory.")
    exit()

# Load the data
df = pd.read_csv(input_file)

# --- 2. Data Cleaning ---
# Remove rows where SMILES is completely missing and ensure everything is a string
df = df.dropna(subset=['SMILES'])
df['SMILES'] = df['SMILES'].astype(str).str.strip()

# --- 3. Define Calculation Logic ---
def calculate_lipinski(smiles):
    """Calculates Ro5 violations; returns None if SMILES is invalid."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Calculate properties
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        
        # Classic Lipinski Rule of 5 (Ro5) cutoffs
        violations = {
            'MW (>500)': mw > 500,
            'LogP (>5)': logp > 5,
            'HBD (>5)': hbd > 5,
            'HBA (>10)': hba > 10
        }
        return violations
    except Exception:
        return None

# --- 4. Processing ---
print("Processing molecules...")
# Apply calculation and drop any that failed to parse (None)
results = df['SMILES'].apply(calculate_lipinski).dropna()
total_valid = len(results)

# Calculate number of violations per molecule
violation_counts = results.apply(lambda x: sum(x.values()))

# Tally 0 and 1 violations
zero_v = sum(violation_counts == 0)
one_v = sum(violation_counts == 1)
total_passed = zero_v + one_v

# --- 5. 1-Violation Breakdown Calculation ---
one_v_details = [res for res in results if sum(res.values()) == 1]
breakdown = {'MW (>500)': 0, 'LogP (>5)': 0, 'HBD (>5)': 0, 'HBA (>10)': 0}

for res in one_v_details:
    for key in breakdown:
        if res[key]:
            breakdown[key] += 1

# --- 6. Summary Statistics & CSV Export ---
summary_df = pd.DataFrame({
    'Metric': [
        'Total Valid Molecules Processed', 
        '--- OVERALL PASS RATES ---',
        'Molecules with 0 Violations', 
        '% of Total Drugs (0 Violations)', 
        'Molecules with 1 Violation', 
        '% of Total Drugs (1 Violation)',
        'Total Molecules Passed (<= 1 Violation)',
        '% Total Passed (<= 1 Violation)',
        '--- 1-VIOLATION BREAKDOWN ---',
        'MW (>500) Violations',
        '% of 1-Violation pool (MW)',
        'LogP (>5) Violations',
        '% of 1-Violation pool (LogP)',
        'HBD (>5) Violations',
        '% of 1-Violation pool (HBD)',
        'HBA (>10) Violations',
        '% of 1-Violation pool (HBA)'
    ],
    'Value': [
        total_valid, 
        '', # Blank row
        zero_v, 
        f"{(zero_v / total_valid * 100):.2f}%" if total_valid > 0 else "0%",
        one_v,
        f"{(one_v / total_valid * 100):.2f}%" if total_valid > 0 else "0%",
        total_passed,
        f"{(total_passed / total_valid * 100):.2f}%" if total_valid > 0 else "0%",
        '', # Blank row
        breakdown['MW (>500)'],
        f"{(breakdown['MW (>500)'] / one_v * 100):.2f}%" if one_v > 0 else "0%",
        breakdown['LogP (>5)'],
        f"{(breakdown['LogP (>5)'] / one_v * 100):.2f}%" if one_v > 0 else "0%",
        breakdown['HBD (>5)'],
        f"{(breakdown['HBD (>5)'] / one_v * 100):.2f}%" if one_v > 0 else "0%",
        breakdown['HBA (>10)'],
        f"{(breakdown['HBA (>10)'] / one_v * 100):.2f}%" if one_v > 0 else "0%"
    ]
})

# Save Summary CSV
summary_df.to_csv(output_csv, index=False)
print(f"Success! Summary saved to {output_csv}")