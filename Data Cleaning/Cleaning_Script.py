import pandas as pd
import numpy as np
import pymongo
import os
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from tqdm import tqdm

# --- Configuration ---
MONGO_URI = "mongodb://root:example@localhost:27017/"
DB_NAME = 'opti_datacollection'
COLLECTION_NAME = 'datacollection'
OUTPUT_DIR = "." # Saved in the same folder as the code

# --- Attrition Tracker ---
tracker = {"Stage": [], "Entries": [], "Molecules": [], "Dropped": [], "% Lost From Start": []}
prev_mol_count = 0
initial_mol_count = 0

def update_tracker(stage, df, id_col='_id'):
    global prev_mol_count, initial_mol_count
    curr_ent = len(df)
    curr_mol = df[id_col].nunique() if not df.empty else 0
    
    if stage == "1_Initial_Extraction":
        initial_mol_count = curr_mol
        
    drop = prev_mol_count - curr_mol if prev_mol_count > 0 else 0
    perc_lost = ((initial_mol_count - curr_mol) / initial_mol_count * 100) if initial_mol_count > 0 else 0
    
    tracker["Stage"].append(stage)
    tracker["Entries"].append(curr_ent)
    tracker["Molecules"].append(curr_mol)
    tracker["Dropped"].append(drop)
    tracker["% Lost From Start"].append(round(perc_lost, 2))
    prev_mol_count = curr_mol

print("--- STARTING STRICT PRESERVE PIPELINE ---")

# --- 1. MongoDB Extraction ---
client = pymongo.MongoClient(MONGO_URI)
collection = client[DB_NAME][COLLECTION_NAME]
query = {"attributes.properties.x_solubility": {"$elemMatch": {"conditions": {"$elemMatch": {"name": "solvent", "value": "InChI=1S/H2O/h1H2"}}}}}
projection = {"_id": 1, "attributes.x_smiles": 1, "attributes.properties.x_solubility": 1}
docs = list(collection.find(query, projection))

flat_data = []
for doc in docs:
    d_id, smi = str(doc.get("_id")), doc.get("attributes", {}).get("x_smiles", "Unknown")
    for entry in doc.get("attributes", {}).get("properties", {}).get("x_solubility", []):
        is_aq, t_val, t_unit = False, None, None
        for cond in entry.get("conditions", []):
            if cond.get("name") == "solvent" and cond.get("value") == "InChI=1S/H2O/h1H2": is_aq = True
            elif cond.get("name") == "temperature": t_val, t_unit = cond.get("value"), cond.get("units", cond.get("unit"))
        if is_aq:
            flat_data.append({"_id": d_id, "Original_SMILES": smi, "Value": entry.get("value"), 
                              "Unit": entry.get("units", entry.get("unit", "")), "Temp_V": t_val, "Temp_U": t_unit})

df = pd.DataFrame(flat_data)
prev_mol_count = df['_id'].nunique()
update_tracker("1_Initial_Extraction", df)

# --- 2. Filtering ---
df = df[df['Unit'].astype(str).str.contains("logS", case=False, na=False)].copy()
df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
df = df.dropna(subset=['Value', 'Temp_V', 'Temp_U'])
df['Temp_V'] = pd.to_numeric(df['Temp_V'], errors='coerce')
df.loc[df['Temp_U'] == 'K', 'Temp_V'] -= 273.15
df = df[(df['Temp_V'] >= 20) & (df['Temp_V'] <= 30)]
update_tracker("2_Metadata_Filter", df)

# --- 3. RDKit Strict Preserve Standardization ---
te = rdMolStandardize.TautomerEnumerator()
te.SetRemoveSp3Stereo(False)   # Protects Wedges/Dashes
te.SetRemoveBondStereo(False)  # Protects Cis/Trans
rdkit_errors = []
valid_rows = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Standardizing (Strict Stereo)"):
    mol = Chem.MolFromSmiles(row['Original_SMILES'])
    if mol and len(Chem.GetMolFrags(mol)) == 1:
        try:
            canon_mol = te.Canonicalize(mol)
            canon_smi = Chem.MolToSmiles(canon_mol, isomericSmiles=True)
        except Exception as e:
            canon_smi = Chem.MolToSmiles(mol, isomericSmiles=True)
            rdkit_errors.append({"_id": row['_id'], "Error": str(e)})
        
        row_dict = row.to_dict()
        row_dict['Std_SMILES'] = canon_smi
        valid_rows.append(row_dict)

df_std = pd.DataFrame(valid_rows)
update_tracker("3_RDKit_Standardize", df_std, id_col='Std_SMILES')

# --- 4. Outlier & Aggregation (Per Isomer) ---
final_results, mapping_data = [], []
d_n2, d_no_out, d_still_high = 0, 0, 0

for std_smi, group in df_std.groupby('Std_SMILES'):
    ids, orig_smi = group['_id'].unique().tolist(), group['Original_SMILES'].unique().tolist()
    if len(ids) > 1:
        mapping_data.append({'Std_SMILES': std_smi, 'Orig_SMILES': " | ".join(orig_smi), 'IDs': " | ".join(ids), 'Count': len(ids)})
    
    vals = group['Value'].values
    if len(vals) == 1:
        final_results.append({"SMILES": std_smi, "LogS_Median": vals[0], "N": 1})
    elif (np.max(vals) - np.min(vals)) <= 0.7:
        final_results.append({"SMILES": std_smi, "LogS_Median": np.median(vals), "N": len(vals)})
    else:
        if len(vals) == 2: 
            d_n2 += 1
        else:
            q1, q3 = np.percentile(vals, [25, 75])
            # CHANGED to 1.5 IQR
            mask = (vals >= q1 - 1.5*(q3-q1)) & (vals <= q3 + 1.5*(q3-q1))
            f_vals = vals[mask]
            
            if len(f_vals) == len(vals): 
                d_no_out += 1
            elif len(f_vals) > 0 and (np.max(f_vals) - np.min(f_vals)) <= 0.7:
                final_results.append({"SMILES": std_smi, "LogS_Median": np.median(f_vals), "N": len(f_vals)})
            else: 
                d_still_high += 1

final_df = pd.DataFrame(final_results)

# Final Tracker Breakdown
final_mol = final_df['SMILES'].nunique()
perc_lost_final = ((initial_mol_count - final_mol) / initial_mol_count * 100) if initial_mol_count > 0 else 0

tracker["Stage"].extend(["4a_Dropped_N2", "4b_Dropped_NoOutlier", "4c_Dropped_StillHigh", "Final"])
tracker["Entries"].extend(["-", "-", "-", len(final_df)])
tracker["Molecules"].extend(["-", "-", "-", final_mol])
tracker["Dropped"].extend([d_n2, d_no_out, d_still_high, 0])
tracker["% Lost From Start"].extend(["-", "-", "-", round(perc_lost_final, 2)])

# --- 5. Final Exports ---
final_df.to_csv(os.path.join(OUTPUT_DIR, "cleaned_solubility.csv"), index=False)
pd.DataFrame(mapping_data).to_csv(os.path.join(OUTPUT_DIR, "duplicate_mapping.csv"), index=False)
pd.DataFrame(tracker).to_csv(os.path.join(OUTPUT_DIR, "attrition_tracker.csv"), index=False)

print(f"--- COMPLETE: Final Molecule Count = {len(final_df)} ---")