import pandas as pd
import numpy as np
import joblib
import warnings
import os
import matplotlib.pyplot as plt
import base64
import shap
from io import BytesIO
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Draw
from sklearn.metrics import pairwise_distances, mean_absolute_error
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Suppress warnings
warnings.filterwarnings('ignore')

# --- Configuration ---
TRAIN_FILE = 'train_data_final.csv'
TEST_FILE = 'test_data_final.csv'
MODEL_FILE = 'trained_model.joblib'
SCALER_FILE = 'scaler_final.joblib'
LINKS_FILE = 'path/to/DrugBankStructurelinks'
PLOTS_DIR = 'Case_Studies_Plot'

if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)

TARGET_ATOMS = ['C', 'H', 'N', 'O', 'S', 'Cl', 'F', 'P', 'I', 'Br', 'Fe', 'Co', 
                'Pt', 'Na', 'Ca', 'Mg', 'B', 'K', 'Al', 'As', 'Au', 'Li', 'Ga', 'Se', 'Si']

# --- Helper Functions ---
def save_and_get_b64(fig, filename):
    filepath = os.path.join(PLOTS_DIR, filename)
    fig.savefig(filepath, format='png', bbox_inches='tight', dpi=150)
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def mol_to_svg(m):
    if not m: return ""
    d = Draw.MolDraw2DSVG(350, 250)
    d.DrawMolecule(m)
    d.FinishDrawing()
    return d.GetDrawingText()

def calculate_atom_counts(mol):
    counts = {el: 0 for el in TARGET_ATOMS}
    counts['C_Aliphatic'] = 0; counts['C_Aromatic'] = 0; counts['O_Aromatic'] = 0
    mol_h = Chem.AddHs(mol)
    for atom in mol_h.GetAtoms():
        sym = atom.GetSymbol()
        if sym in counts: counts[sym] += 1
        if sym == 'C':
            if atom.GetIsAromatic(): counts['C_Aromatic'] += 1
            else: counts['C_Aliphatic'] += 1
        elif sym == 'O' and atom.GetIsAromatic(): counts['O_Aromatic'] += 1
    return counts

def calculate_bond_counts(mol):
    counts = {'single_bonds': 0, 'double_bonds': 0, 'triple_bonds': 0, 'aromatic_bonds': 0}
    for bond in mol.GetBonds():
        b_type = bond.GetBondType()
        if b_type == Chem.rdchem.BondType.SINGLE: counts['single_bonds'] += 1
        elif b_type == Chem.rdchem.BondType.DOUBLE: counts['double_bonds'] += 1
        elif b_type == Chem.rdchem.BondType.TRIPLE: counts['triple_bonds'] += 1
        elif b_type == Chem.rdchem.BondType.AROMATIC: counts['aromatic_bonds'] += 1
    return counts

def calculate_polar_bond_counts(mol):
    mol_h = Chem.AddHs(mol)
    counts = {'OH_bonds': 0, 'NH_bonds': 0, 'SH_bonds': 0, 'CO_bonds': 0, 'CN_bonds': 0, 'C_Halogen_bonds': 0}
    halogens = {9, 17, 35, 53} 
    for bond in mol_h.GetBonds():
        a1, a2 = bond.GetBeginAtom().GetAtomicNum(), bond.GetEndAtom().GetAtomicNum()
        nums = tuple(sorted([a1, a2]))
        if nums == (1, 8): counts['OH_bonds'] += 1
        elif nums == (1, 7): counts['NH_bonds'] += 1
        elif nums == (1, 16): counts['SH_bonds'] += 1
        elif nums == (6, 8): counts['CO_bonds'] += 1
        elif nums == (6, 7): counts['CN_bonds'] += 1
        elif nums[0] == 6 and nums[1] in halogens: counts['C_Halogen_bonds'] += 1
    return counts

def calculate_ring_counts(mol):
    ring_info = mol.GetRingInfo()
    counts = {'ring_size_4': 0, 'ring_size_5': 0, 'ring_size_6': 0, 'ring_size_7': 0, 'ring_size_8': 0, 'macrocycles': 0}
    for ring in ring_info.AtomRings():
        size = len(ring)
        if size == 4: counts['ring_size_4'] += 1
        elif size == 5: counts['ring_size_5'] += 1
        elif size == 6: counts['ring_size_6'] += 1
        elif size == 7: counts['ring_size_7'] += 1
        elif size == 8: counts['ring_size_8'] += 1
        elif size > 8: counts['macrocycles'] += 1
    return counts

def get_inchikey(smi):
    try:
        m = Chem.MolFromSmiles(str(smi))
        return Chem.MolToInchiKey(m) if m else None
    except:
        return None

def main():
    print("Loading initial data and models...")
    test_df = pd.read_csv(TEST_FILE)
    train_df = pd.read_csv(TRAIN_FILE)
    links_df = pd.read_csv(LINKS_FILE)
    
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    
    features = [c for c in train_df.columns if c not in ['SMILES', 'LogS_Median']]
    
    # --- STAGE 1: Random Sample of 10 Drugs ---
    print("Mapping drugs and selecting 10 random targets (Seed 42)...")
    test_df['y_pred_global'] = model.predict(scaler.transform(test_df[features].values))
    
    key_map = dict(zip(links_df['InChIKey'], links_df['Name']))
    test_df['InChIKey'] = test_df['SMILES'].apply(get_inchikey)
    test_df['Drug_Name'] = test_df['InChIKey'].map(key_map)
    
    # Filter to identified drugs
    drug_df = test_df[test_df['Drug_Name'].notna()].copy()
    
    # Draw the random 10 first
    random_10 = drug_df.sample(n=10, random_state=42).copy()
    
    # --- STAGE 2: Pick 3 from the 10 based on Error ---
    print("Selecting Best, Median, and Worst from the random 10...")
    random_10['abs_error'] = (random_10['LogS_Median'] - random_10['y_pred_global']).abs()
    random_10 = random_10.sort_values(by='abs_error').reset_index(drop=True)
    
    # Extract the extremes from the 10
    best = random_10.iloc[[0]].copy(); best['Perf_Label'] = "BEST of Sample (Smallest Error)"
    worst = random_10.iloc[[-1]].copy(); worst['Perf_Label'] = "WORST of Sample (Largest Error)"
    median = random_10.iloc[[4]].copy(); median['Perf_Label'] = "MEDIAN of Sample" 
    
    selected_drugs = pd.concat([best, median, worst]).reset_index(drop=True)
    
    print(f"Sub-selection complete. Best: {selected_drugs.iloc[0]['Drug_Name']}, "
          f"Median: {selected_drugs.iloc[1]['Drug_Name']}, "
          f"Worst: {selected_drugs.iloc[2]['Drug_Name']}")

    # --- Setup Global Similarity Space (MACCS / Cont) ---
    maccs_feats = [f for f in features if f.startswith('MACCS_')]
    cont_feats = [f for f in features if not f.startswith('MACCS_')]
    
    sim_scaler = StandardScaler()
    train_cont_sc = sim_scaler.fit_transform(train_df[cont_feats].values)
    nbrs = NearestNeighbors(n_neighbors=5, metric='euclidean').fit(train_cont_sc)
    train_maccs = train_df[maccs_feats].values.astype(bool)

    # Pre-calculate whole test set similarity for the "Historical Metrics"
    test_cont_sc = sim_scaler.transform(test_df[cont_feats].values)
    dists_t, _ = nbrs.kneighbors(test_cont_sc)
    test_normalized_dists = dists_t / np.sqrt(len(cont_feats))
    test_sim_desc = np.mean(1 / (1 + test_normalized_dists), axis=1)
    
    test_maccs_vals = test_df[maccs_feats].values.astype(bool)
    jdists_t = pairwise_distances(test_maccs_vals, train_maccs, metric='jaccard')
    tsims_t = 1 - jdists_t
    top5_t = np.sort(tsims_t, axis=1)[:, -5:]
    test_sim_fp = np.mean(top5_t, axis=1)
    test_df['sim_avg'] = (test_sim_desc + test_sim_fp) / 2

    # --- NEW: Calculate the Overall Test Set MAE by Similarity Bin ---
    bins = [0, 0.4, 0.6, 0.7, 0.8, 0.9, 1.01]
    labels = ["<40%", "40-60%", "60-70%", "70-80%", "80-90%", "90%+"]
    test_df['Sim_Bin'] = pd.cut(test_df['sim_avg'], bins=bins, labels=labels, right=False)
    test_df['abs_error_global'] = np.abs(test_df['LogS_Median'] - test_df['y_pred_global'])
    
    # Create a dictionary mapping the Bin Label to its Mean Absolute Error
    bin_mae_lookup = test_df.groupby('Sim_Bin', observed=False)['abs_error_global'].mean().to_dict()

    # Initialize Explainer
    explainer = shap.TreeExplainer(model)
    expected_val = explainer.expected_value[0] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value

    html_blocks = []

    # --- Build Summary Table for all 10 Random Drugs ---
    summary_rows = ""
    for _, r in random_10.iterrows():
        row_style = ""
        if r['Drug_Name'] == best['Drug_Name'].values[0]: row_style = "background-color: #d5f5e3; font-weight: bold;" 
        elif r['Drug_Name'] == worst['Drug_Name'].values[0]: row_style = "background-color: #fadbd8; font-weight: bold;" 
        elif r['Drug_Name'] == median['Drug_Name'].values[0]: row_style = "background-color: #fdebd0; font-weight: bold;" 
            
        summary_rows += f"""
        <tr style="border-bottom: 1px solid #eee; {row_style}">
            <td style="padding: 10px;">{r['Drug_Name']}</td>
            <td style="padding: 10px;">{r['LogS_Median']:.2f}</td>
            <td style="padding: 10px;">{r['y_pred_global']:.2f}</td>
            <td style="padding: 10px;">{r['abs_error']:.3f}</td>
        </tr>"""

    summary_html = f"""
    <div class="container" style="background: white; padding: 25px; border-radius: 10px; box-shadow: 0 4px 10px rgba(0,0,0,0.05); margin-bottom: 50px; border-top: 5px solid #2c3e50;">
        <h2 style="margin-top: 0; color: #2c3e50;">Random Subset Summary (n=10)</h2>
        <p style="color: #666; margin-bottom: 20px;">The 10 randomly selected approved drugs. The Best (Green), Median (Orange), and Worst (Red) performers have been extracted for detailed case studies below.</p>
        <table style="width: 100%; border-collapse: collapse; text-align: left;">
            <tr style="background: #2c3e50; color: white;">
                <th style="padding: 12px;">Drug Name</th>
                <th style="padding: 12px;">Actual LogS</th>
                <th style="padding: 12px;">Predicted LogS</th>
                <th style="padding: 12px;">Absolute Error</th>
            </tr>
            {summary_rows}
        </table>
    </div>
    """
    html_blocks.append(summary_html)


    # --- 3. Process the 3 selected examples ---
    for idx, row in selected_drugs.iterrows():
        drug_name = row['Drug_Name']
        perf_label = row['Perf_Label']
        user_smiles = row['SMILES']
        actual_logs = row['LogS_Median']
        
        print(f"Generating details for: {drug_name}...")
        mol = Chem.MolFromSmiles(user_smiles)
        
        # Re-featurize for this specific instance
        rdkit_vals = {desc_func[0]: desc_func[1](mol) for desc_func in Descriptors.descList}
        maccs_fp = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
        maccs_vals = {f"MACCS_{i}": list(maccs_fp)[i] for i in range(1, 167)}
        atom_c = calculate_atom_counts(mol)
        bond_c = calculate_bond_counts(mol)
        polar_c = calculate_polar_bond_counts(mol)
        ring_c = calculate_ring_counts(mol)
        
        all_calcs = {**rdkit_vals, **maccs_vals, **atom_c, **bond_c, **polar_c, **ring_c}
        input_df = pd.DataFrame([all_calcs], columns=features).fillna(0)
        
        # Scaled input for prediction & SHAP
        X_input_sc = scaler.transform(input_df.values)
        pred_logs = model.predict(X_input_sc)[0]
        
        # Local Similarity calculation 
        input_cont_sc = sim_scaler.transform(input_df[cont_feats].values)
        dists, _ = nbrs.kneighbors(input_cont_sc)
        local_norm_dist = dists[0] / np.sqrt(len(cont_feats))
        sim_desc = np.mean(1 / (1 + local_norm_dist))
        
        input_maccs_vals = input_df[maccs_feats].values.astype(bool)
        tanimoto_sims = 1 - pairwise_distances(input_maccs_vals, train_maccs, metric='jaccard')[0]
        sim_fp = np.mean(np.sort(tanimoto_sims)[-5:])
        
        sim_avg_raw = (sim_desc + sim_fp) / 2
        sim_avg_pct = sim_avg_raw * 100
        
        # --- NEW: Lookup Expected Bin MAE ---
        local_bin = pd.cut([sim_avg_raw], bins=bins, labels=labels, right=False)[0]
        expected_mae = bin_mae_lookup.get(local_bin, float('nan'))
        expected_mae_text = f"{expected_mae:.3f}" if pd.notna(expected_mae) else "N/A"

        # SHAP Waterfall
        shap_vals = explainer.shap_values(X_input_sc)
        exp = shap.Explanation(values=shap_vals[0], base_values=expected_val, data=input_df.iloc[0].values, feature_names=features)
        fig = plt.figure(figsize=(9, 6))
        shap.plots.waterfall(exp, max_display=10, show=False)
        
        # Modify the text objects in the waterfall plot
        ax = plt.gca()
        for text in ax.texts:
            text.set_color('black')
            text.set_fontweight('bold')
            text.set_fontsize(11)
        for tick in ax.get_yticklabels():
            tick.set_color('black')
            tick.set_fontweight('bold')
            tick.set_fontsize(11)
        for tick in ax.get_xticklabels():
            tick.set_color('black')
            tick.set_fontweight('bold')
            tick.set_fontsize(11)

        plt.tight_layout()
        
        safe_name = "".join([c for c in drug_name if c.isalnum() or c in " -_"]).strip()
        b64_wf = save_and_get_b64(fig, f"shap_waterfall_{safe_name}.png")
        
        svg = mol_to_svg(mol)
        svg_b64 = base64.b64encode(svg.encode('utf-8')).decode('utf-8')
        
        # HTML Block
        html_block = f"""
        <div class="container" style="background: white; padding: 25px; border-radius: 10px; box-shadow: 0 4px 10px rgba(0,0,0,0.05); margin-bottom: 50px; border-top: 5px solid #3498db;">
            <div style="background: #ebf5fb; padding: 12px; border-radius: 5px; margin-bottom: 20px; text-align: center; font-weight: bold; color: #2980b9; font-size: 1.2em;">
                {perf_label}
            </div>
            
            <div style="display: flex; flex-wrap: wrap; gap: 20px; align-items: flex-start;">
                
                <div style="flex: 1; min-width: 350px; text-align: center; border-right: 1px solid #eee; padding-right: 20px;">
                    <h2 style="margin-top: 0;">{drug_name}</h2>
                    <div style="background: white; padding: 10px; border-radius: 8px;">
                        {svg}
                    </div>
                    <a href="data:image/svg+xml;base64,{svg_b64}" download="{safe_name}_structure.svg" 
                       style="display: inline-block; margin-top: 15px; background: #2ecc71; color: white; padding: 8px 15px; border-radius: 5px; text-decoration: none; font-weight: bold; font-size: 0.9em;">
                        ⬇ Download Molecule (SVG)
                    </a>
                    <div style="font-family: monospace; font-size: 0.8em; color: #666; margin-top: 20px; word-break: break-all; background: #f9f9f9; padding: 10px; border: 1px solid #eee;">
                        <strong>SMILES:</strong> {user_smiles}
                    </div>
                </div>
                
                <div style="flex: 1; min-width: 300px;">
                    <h3 style="margin-top: 0; color: #2c3e50;">Prediction Metrics</h3>
                    <div style="font-size: 1.2em; color: #7f8c8d; margin-bottom: 5px;">Predicted LogS</div>
                    <div style="font-size: 2.5em; font-weight: bold; color: #3498db; margin-bottom: 15px;">{pred_logs:.2f}</div>
                    
                    <div style="font-size: 1.1em; color: #333; margin-top: 5px;"><b>Actual LogS:</b> {actual_logs:.2f}</div>
                    <div style="font-size: 1.1em; color: #333; margin-top: 5px;"><b>Absolute Error:</b> {row['abs_error']:.3f}</div>
                    
                    <hr style="border: 0; border-top: 1px solid #eee; margin: 15px 0;">
                    
                    <div style="font-size: 1.1em; color: #333; margin-top: 5px;"><b>Similarity to Training Data:</b> {sim_avg_pct:.1f}%</div>
                    <div style="font-size: 1.1em; color: #333; margin-top: 5px;"><b>Similarity Bin:</b> {local_bin}</div>
                    <div style="font-size: 1.1em; color: #333; margin-top: 5px;"><b>Expected Bin MAE:</b> {expected_mae_text}</div>
                </div>
            </div>
            
            <div style="margin-top: 40px;">
                <h3 style="color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 10px;">Explainability (Top 10 Features)</h3>
                <p style="color: #555; font-size: 0.95em;">The plot below illustrates how each individual chemical feature drove the model's prediction away from the baseline expected LogS.</p>
                <img src="data:image/png;base64,{b64_wf}" style="width: 100%; max-width: 900px; display: block; margin: 0 auto; border: 1px solid #ddd; border-radius: 5px; padding: 10px;">
            </div>
        </div>
        """
        html_blocks.append(html_block)

    # Save outputs
    print("Saving HTML Report...")
    with open("Random_Sample_Performance_Report.html", "w", encoding='utf-8') as f:
        f.write(f"<html><head><meta charset='UTF-8'></head><body style='background:#f0f3f5; padding:40px; font-family:sans-serif;'>{''.join(html_blocks)}</body></html>")
    
    print("Done! Report generated successfully.")

if __name__ == "__main__":
    main()
    