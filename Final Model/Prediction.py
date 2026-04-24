import pandas as pd
import numpy as np
import os
import joblib
import warnings
import matplotlib.pyplot as plt
import base64
import shap
import difflib
from io import BytesIO
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Draw
from sklearn.metrics import pairwise_distances, mean_squared_error, mean_absolute_error
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, normalize

# Suppress warnings for cleaner terminal output
warnings.filterwarnings('ignore')

# --- Configuration ---
TRAIN_FILE = 'train_data_final.csv'
TEST_FILE = 'test_data_final.csv'
MODEL_FILE = 'trained_model.joblib'
SCALER_FILE = 'scaler_final.joblib'
LINKS_FILE = 'path/to/DrugBankStructurelinks'

TARGET_ATOMS = ['C', 'H', 'N', 'O', 'S', 'Cl', 'F', 'P', 'I', 'Br', 'Fe', 'Co', 
                'Pt', 'Na', 'Ca', 'Mg', 'B', 'K', 'Al', 'As', 'Au', 'Li', 'Ga', 'Se', 'Si']

def get_b64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=120)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def mol_to_svg(m):
    if not m: return ""
    d = Draw.MolDraw2DSVG(350, 250)
    d.DrawMolecule(m)
    d.FinishDrawing()
    return d.GetDrawingText()

# --- Featurization Functions (Matched to Training Script) ---
def calculate_atom_counts(mol):
    if mol is None: return None
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
    if mol is None: return None
    counts = {'single_bonds': 0, 'double_bonds': 0, 'triple_bonds': 0, 'aromatic_bonds': 0}
    for bond in mol.GetBonds():
        b_type = bond.GetBondType()
        if b_type == Chem.rdchem.BondType.SINGLE: counts['single_bonds'] += 1
        elif b_type == Chem.rdchem.BondType.DOUBLE: counts['double_bonds'] += 1
        elif b_type == Chem.rdchem.BondType.TRIPLE: counts['triple_bonds'] += 1
        elif b_type == Chem.rdchem.BondType.AROMATIC: counts['aromatic_bonds'] += 1
    return counts

def calculate_polar_bond_counts(mol):
    if mol is None: return None
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
    if mol is None: return None
    ring_info = mol.GetRingInfo()
    counts = {'ring_size_4': 0, 'ring_size_5': 0, 'ring_size_6': 0, 'ring_size_7': 0, 'ring_size_8': 0, 'macrocycles': 0}
    for ring in ring_info.AtomRings():
        size = len(ring)
        if 4 <= size <= 8: counts[f'ring_size_{size}'] += 1
        elif size > 8: counts['macrocycles'] += 1
    return counts

def main():
    print("\n" + "="*50)
    print("💧 SOLUBILITY PREDICTOR")
    print("="*50)
    
    # 1. Interactive Inputs
    user_smiles = input("Enter SMILES string: ").strip()
    mol = Chem.MolFromSmiles(user_smiles)
    if not mol:
        print("❌ Error: Invalid SMILES string. Please try again.")
        return

    # SHAP Feature Count
    shap_max_input = input("How many features to show in SHAP plot? [Default: 10]: ").strip()
    try:
        shap_max = int(shap_max_input) if shap_max_input else 10
    except ValueError:
        print("⚠️ Invalid number, defaulting to 10.")
        shap_max = 10

    # 2. Load Models & Reference Data
    print("\n⏳ Loading models and reference data...")
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)
    links_df = pd.read_csv(LINKS_FILE)
    
    # Safely get feature names from the training dataframe
    features = [c for c in train_df.columns if c not in ['SMILES', 'LogS_Median']]

    # 3. Interactive Multi-Feature Lookup
    features_to_lookup = []
    feature_input = input("Enter feature names to look up (comma-separated, or press Enter to skip): ").strip()
    
    if feature_input:
        raw_list = [f.strip() for f in feature_input.split(',')]
        for feat in raw_list:
            if not feat: continue
            if feat in features:
                features_to_lookup.append(feat)
            else:
                close_matches = difflib.get_close_matches(feat, features, n=3, cutoff=0.4)
                if close_matches:
                    match = close_matches[0]
                    confirm = input(f"⚠️ Feature '{feat}' not found. Did you mean '{match}'? (y/n): ").strip().lower()
                    if confirm == 'y':
                        features_to_lookup.append(match)
                    else:
                        print(f"Skipping '{feat}'.")
                else:
                    print(f"❌ Feature '{feat}' not found and no close matches detected. Skipping.")

    # 4. Check if it's an Approved Drug
    inchikey = Chem.MolToInchiKey(mol)
    drug_name = "Novel Compound"
    is_drug = False
    match = links_df[links_df['InChIKey'] == inchikey]
    if not match.empty:
        drug_name = match.iloc[0]['Name']
        is_drug = True
        print(f"✅ Matched in DrugBank: {drug_name}")

    # 5. Featurize the Input
    print("⚙️  Calculating RDKit, MACCS, and Custom Features...")
    rdkit_vals = {desc_func[0]: desc_func[1](mol) for desc_func in Descriptors.descList}
    maccs_fp = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
    maccs_vals = {f"MACCS_{i}": list(maccs_fp)[i] for i in range(1, 167)}
    
    atom_c = calculate_atom_counts(mol)
    bond_c = calculate_bond_counts(mol)
    polar_c = calculate_polar_bond_counts(mol)
    ring_c = calculate_ring_counts(mol)
    
    all_calcs = {**rdkit_vals, **maccs_vals, **atom_c, **bond_c, **polar_c, **ring_c}
    
    # This magically ignores any features dropped during data cleaning
    input_df = pd.DataFrame([all_calcs], columns=features).fillna(0) 

    # 6. Predict
    X_input_sc = scaler.transform(input_df)
    pred_logs = model.predict(X_input_sc)[0]

    # 7. Calculate Similarity (StandardScaler + Normalization)
    print("🔍 Calculating Applicability Domain (Similarity)...")
    maccs_feats = [f for f in features if f.startswith('MACCS_')]
    cont_feats = [f for f in features if not f.startswith('MACCS_')]

    sim_scaler = StandardScaler()
    
    # Scale and normalize continuous features
    train_cont_sc = normalize(sim_scaler.fit_transform(train_df[cont_feats]))
    input_cont_sc = normalize(sim_scaler.transform(input_df[cont_feats]))
    
    nbrs = NearestNeighbors(n_neighbors=5, metric='euclidean').fit(train_cont_sc)
    dists, _ = nbrs.kneighbors(input_cont_sc)
    sim_desc = np.mean(1 / (1 + dists[0]))

    train_maccs = train_df[maccs_feats].values
    input_maccs = input_df[maccs_feats].values
    jaccard_dists = pairwise_distances(input_maccs, train_maccs, metric='jaccard')
    tanimoto_sims = 1 - jaccard_dists[0]
    top5_sims = np.sort(tanimoto_sims)[-5:]
    sim_fp = np.mean(top5_sims)
    
    sim_avg = (sim_desc + sim_fp) / 2

    # Confidence Bracketing
    if sim_avg > 0.8:
        conf_label = "High (>80%)"
        conf_color = "#2ecc71"
        test_mask = (test_df['sim_avg'] > 0.8) if 'sim_avg' in test_df.columns else None
    elif sim_avg >= 0.6:
        conf_label = "Medium (60-80%)"
        conf_color = "#f39c12"
        test_mask = ((test_df['sim_avg'] >= 0.6) & (test_df['sim_avg'] <= 0.8)) if 'sim_avg' in test_df.columns else None
    else:
        conf_label = "Low (<60% - Extrapolation)"
        conf_color = "#e74c3c"
        test_mask = (test_df['sim_avg'] < 0.6) if 'sim_avg' in test_df.columns else None

    hist_stats = {"RMSE": "N/A", "MAE": "N/A", "Pct07": "N/A", "N": 0}
    if 'sim_avg' not in test_df.columns:
        test_cont_sc = normalize(sim_scaler.transform(test_df[cont_feats]))
        dists_t, _ = nbrs.kneighbors(test_cont_sc)
        test_sim_desc = np.mean(1 / (1 + dists_t), axis=1)
        test_maccs = test_df[maccs_feats].values
        jdists_t = pairwise_distances(test_maccs, train_maccs, metric='jaccard')
        tsims_t = 1 - jdists_t
        top5_t = np.sort(tsims_t, axis=1)[:, -5:]
        test_sim_fp = np.mean(top5_t, axis=1)
        test_df['sim_avg'] = (test_sim_desc + test_sim_fp) / 2
        
        if sim_avg > 0.8: test_mask = test_df['sim_avg'] > 0.8
        elif sim_avg >= 0.6: test_mask = (test_df['sim_avg'] >= 0.6) & (test_df['sim_avg'] <= 0.8)
        else: test_mask = test_df['sim_avg'] < 0.6

    test_df['y_pred'] = model.predict(scaler.transform(test_df[features]))
    bracket_df = test_df[test_mask]
    
    if len(bracket_df) > 0:
        hist_stats["N"] = len(bracket_df)
        hist_stats["RMSE"] = f"{np.sqrt(mean_squared_error(bracket_df['LogS_Median'], bracket_df['y_pred'])):.2f}"
        hist_stats["MAE"] = f"{mean_absolute_error(bracket_df['LogS_Median'], bracket_df['y_pred']):.2f}"
        hist_stats["Pct07"] = f"{np.mean(np.abs(bracket_df['LogS_Median'] - bracket_df['y_pred']) <= 0.7) * 100:.1f}%"

    # 8. Explainability & SHAP Logic
    print("📊 Generating SHAP Waterfall plot and calculating feature importance...")
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_input_sc)
    expected_val = explainer.expected_value[0] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value
    
    # Process multi-feature lookup stats
    lookup_html = ""
    if features_to_lookup:
        shap_vals_array = shap_vals[0]
        abs_shap = np.abs(shap_vals_array)
        ranked_indices = np.argsort(-abs_shap)
        
        for actual_feature in features_to_lookup:
            idx = features.index(actual_feature)
            rank = np.where(ranked_indices == idx)[0][0] + 1
            f_val = input_df.iloc[0][actual_feature]
            s_val = shap_vals_array[idx]
            
            lookup_html += f"""
            <div class="card" style="border-left-color: #9b59b6; margin-top: 20px;">
                <h3 style="margin-bottom: 5px;">🔍 Specific Feature Analysis: <code>{actual_feature}</code></h3>
                <ul style="font-size: 1.05em;">
                    <li><b>Molecule's Input Value:</b> {f_val:.4f}</li>
                    <li><b>SHAP Impact on Solubility:</b> <span style="color: {'#e74c3c' if s_val > 0 else '#2980b9'}; font-weight: bold;">{s_val:+.4f}</span> <i>(positive increases predicted solubility, negative decreases it)</i></li>
                    <li><b>Overall Importance Rank:</b> #{rank} out of {len(features)} total features</li>
                </ul>
            </div>
            """

    # Generate Waterfall Image
    exp = shap.Explanation(values=shap_vals[0], base_values=expected_val, data=input_df.iloc[0].values, feature_names=features)
    fig_waterfall = plt.figure(figsize=(8, 6))
    shap.plots.waterfall(exp, max_display=shap_max, show=False)
    plt.title(f"SHAP Feature Impacts: {drug_name}", fontsize=14)
    plt.tight_layout()
    b64_wf = get_b64(fig_waterfall)
    svg = mol_to_svg(mol)

    # 9. HTML Report Generation
    safe_name = drug_name.replace(" ", "_").replace("/", "_")
    html_out = f"Prediction_Report_{safe_name}.html"
    
    html = f"""
    <html><head><meta charset="UTF-8"><style>
        body {{ font-family: 'Segoe UI', sans-serif; background: #f4f7f9; color: #333; margin: 0; padding: 40px; display: flex; justify-content: center; }}
        .container {{ background: white; padding: 30px; border-radius: 12px; box-shadow: 0 8px 20px rgba(0,0,0,0.1); max-width: 800px; width: 100%; }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; text-align: center; }}
        .header-box {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; background: #fcfcfc; border: 1px solid #eee; padding: 20px; border-radius: 8px; }}
        .mol-drawing {{ background: white; text-align: center; }}
        .pred-box {{ text-align: right; }}
        .pred-val {{ font-size: 3em; font-weight: bold; color: #2980b9; }}
        .info-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 30px; }}
        .card {{ background: #f9f9f9; padding: 20px; border-radius: 8px; border-left: 5px solid #bdc3c7; }}
        .card.conf {{ border-left-color: {conf_color}; }}
        h3 {{ margin-top: 0; color: #2c3e50; font-size: 1.1em; }}
        ul {{ padding-left: 20px; margin-bottom: 0; }}
        li {{ margin-bottom: 8px; }}
        .shap-img {{ width: 100%; border: 1px solid #eee; border-radius: 8px; margin-top: 10px; }}
    </style></head><body>

    <div class="container">
        <h1>Solubility Prediction Report</h1>
        
        <div class="header-box">
            <div class="mol-drawing">
                <h2 style="margin-top:0;">{drug_name}</h2>
                {svg}
            </div>
            <div class="pred-box">
                <div style="font-size: 1.2em; color: #7f8c8d;">Predicted LogS</div>
                <div class="pred-val">{pred_logs:.2f}</div>
            </div>
        </div>

        <div class="info-grid">
            <div class="card conf">
                <h3>Applicability Domain (Similarity)</h3>
                <p style="margin-top:0; color:{conf_color}; font-weight:bold; font-size:1.2em;">Confidence: {conf_label}</p>
                <ul>
                    <li><b>Average Similarity:</b> {sim_avg*100:.1f}%</li>
                    <li><b>MACCS Fingerprint:</b> {sim_fp*100:.1f}%</li>
                    <li><b>Continuous Descriptors:</b> {sim_desc*100:.1f}%</li>
                </ul>
            </div>
            
            <div class="card">
                <h3>Historical Metrics For This Region</h3>
                <p style="margin-top:0; font-size:0.9em; color:#7f8c8d;">Based on {hist_stats['N']} similar molecules from the historical test set.</p>
                <ul>
                    <li><b>Expected RMSE:</b> {hist_stats['RMSE']}</li>
                    <li><b>Expected MAE:</b> {hist_stats['MAE']}</li>
                    <li><b>Historically within 0.7 units:</b> {hist_stats['Pct07']}</li>
                </ul>
            </div>
        </div>

        <h3>Prediction Explainability (Top {shap_max} Features)</h3>
        <p style="font-size: 0.9em; color: #666;">This SHAP Waterfall plot shows exactly how the top influential features for this specific molecule pushed the predicted solubility either up (red) or down (blue) from the baseline dataset average.</p>
        <img src="data:image/png;base64,{b64_wf}" class="shap-img">
        
        {lookup_html}
    </div>

    </body></html>
    """
    with open(html_out, "w", encoding='utf-8') as f: f.write(html)
    print(f"✅ Detailed report safely generated: {html_out}\n")

if __name__ == "__main__":
    main()