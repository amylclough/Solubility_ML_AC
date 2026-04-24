import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, MACCSkeys

# ==========================================
# CONFIGURATION
# ==========================================
STRUCTURE_LINKS = 'path/to/DrugBankStructurelinks'
AQSOLDB_FILE = 'path/to/AqSolDBc'
TRAIN_FILE = 'train_data_final.csv'
TEST_FILE = 'test_data_final.csv'
MODEL_FILE = 'trained_model.joblib'
SCALER_FILE = 'scaler_final.joblib'
HTML_OUT = 'External_Validation_Report.html'
PLOTS_DIR = 'external_plots'

if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)

# Featurization Targets
TARGET_ATOMS = ['C', 'H', 'N', 'O', 'S', 'Cl', 'F', 'P', 'I', 'Br', 'Fe', 'Co', 
                'Pt', 'Na', 'Ca', 'Mg', 'B', 'K', 'Al', 'As', 'Au', 'Li', 'Ga', 'Se', 'Si']

# ==========================================
# 1. FEATURIZATION FUNCTIONS
# ==========================================
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
        if 4 <= size <= 8: counts[f'ring_size_{size}'] += 1
        elif size > 8: counts['macrocycles'] += 1
    return counts

# ==========================================
# 2. UTILITY FUNCTIONS
# ==========================================
def save_and_get_b64(fig, filename):
    filepath = os.path.join(PLOTS_DIR, filename)
    fig.savefig(filepath, format='png', bbox_inches='tight', dpi=120)
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=120)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def get_inchikey(smiles):
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        return Chem.MolToInchiKey(mol) if mol else None
    except: return None

def calc_5_metrics(y_true, y_pred):
    if len(y_true) < 2: return {"R2": np.nan, "RMSE": np.nan, "MAE": np.nan, "Pct_07": np.nan, "Pct_10": np.nan}
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "Pct_07": np.mean(np.abs(y_true - y_pred) <= 0.7) * 100,
        "Pct_10": np.mean(np.abs(y_true - y_pred) <= 1.0) * 100
    }

# ==========================================
# 3. MAIN PIPELINE
# ==========================================
def main():
    sns.set_theme(style="whitegrid")
    stats_tracker = {}

    print("\n--- STEP 1: Ingestion & Curation ---")
    struct_df = pd.read_csv(STRUCTURE_LINKS).dropna(subset=['SMILES'])
    stats_tracker['original_structure_links'] = len(struct_df)
    
    curated_records = []
    for _, row in tqdm(struct_df.iterrows(), total=len(struct_df), desc="Curating Structure Links"):
        smiles = str(row['SMILES']).strip()
        if '.' in smiles: continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: continue
        
        mw = Descriptors.MolWt(mol)
        if mw > 700 or mw < 50: continue
        
        unsupported = any(atom.GetSymbol() not in TARGET_ATOMS for atom in mol.GetAtoms())
        if unsupported: continue
        
        curated_records.append({'Name': row.get('Name', 'Unknown'), 'SMILES': smiles, 'InChIKey': Chem.MolToInchiKey(mol)})
    
    curated_df = pd.DataFrame(curated_records)
    stats_tracker['curated_drugs'] = len(curated_df)

    print("\n--- STEP 2: Train/Test Overlap Check ---")
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)
    
    train_keys = set([get_inchikey(s) for s in train_df['SMILES']])
    test_keys = set([get_inchikey(s) for s in test_df['SMILES']])
    known_keys = train_keys.union(test_keys)
    
    # Calculate EXACTLY how many curated external drugs overlap with your models
    overlap_train = curated_df['InChIKey'].isin(train_keys).sum()
    overlap_test = curated_df['InChIKey'].isin(test_keys).sum()
    
    stats_tracker['overlap_train'] = overlap_train
    stats_tracker['overlap_test'] = overlap_test
    stats_tracker['overlap_total'] = overlap_train + overlap_test
    
    # Isolate strictly external drugs
    ext_isolated_df = curated_df[~curated_df['InChIKey'].isin(known_keys)].copy()
    stats_tracker['isolated_external'] = len(ext_isolated_df)

    print("\n--- STEP 3: AqSolDB Cross-Reference ---")
    aq_df = pd.read_csv(AQSOLDB_FILE)
    aq_df['InChIKey'] = [get_inchikey(s) for s in tqdm(aq_df['SMILES'], desc="Hashing AqSolDB")]
    aq_lookup = aq_df.dropna(subset=['InChIKey']).set_index('InChIKey')['ExperimentalLogS'].to_dict()
    
    ext_isolated_df['LogS_Median'] = ext_isolated_df['InChIKey'].map(aq_lookup)
    ext_final_df = ext_isolated_df.dropna(subset=['LogS_Median']).copy()
    stats_tracker['final_aqsol_matched'] = len(ext_final_df)
    
    if len(ext_final_df) == 0:
        print("\nCRITICAL: No external drugs were found in AqSolDB. Pipeline terminating.")
        return

    print("\n--- STEP 4: Featurization ---")
    features_ordered = [c for c in train_df.columns if c not in ['SMILES', 'LogS_Median']]
    rdkit_names = [d[0] for d in Descriptors.descList]
    
    feat_results = []
    valid_ext_smiles = []
    for sm in tqdm(ext_final_df['SMILES'], desc="Calculating Features"):
        mol = Chem.MolFromSmiles(sm)
        if mol:
            rdkit_vals = [desc_func(mol) for _, desc_func in Descriptors.descList]
            maccs_vals = list(rdMolDescriptors.GetMACCSKeysFingerprint(mol))[1:] 
            custom_vals = list(calculate_atom_counts(mol).values()) + \
                          list(calculate_bond_counts(mol).values()) + \
                          list(calculate_polar_bond_counts(mol).values()) + \
                          list(calculate_ring_counts(mol).values())
            
            # Map everything to a dictionary so we can extract exactly what the model needs
            all_feats = dict(zip(
                rdkit_names + [f"MACCS_{i}" for i in range(1, 167)] + 
                list(calculate_atom_counts(mol).keys()) + list(calculate_bond_counts(mol).keys()) +
                list(calculate_polar_bond_counts(mol).keys()) + list(calculate_ring_counts(mol).keys()),
                rdkit_vals + maccs_vals + custom_vals
            ))
            
            feat_results.append([all_feats.get(f, 0) for f in features_ordered])
            valid_ext_smiles.append(sm)

    ext_feat_df = pd.DataFrame(feat_results, columns=features_ordered)
    ext_feat_df['SMILES'] = valid_ext_smiles
    ext_df = pd.merge(ext_final_df[['Name', 'SMILES', 'LogS_Median', 'InChIKey']], ext_feat_df, on='SMILES')

    print("\n--- STEP 5: Machine Learning & Similarity ---")
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    
    X_ext_sc = scaler.transform(ext_df[features_ordered].values)
    ext_df['y_pred'] = model.predict(X_ext_sc)
    ext_df['abs_err'] = np.abs(ext_df['LogS_Median'] - ext_df['y_pred'])

    # Similarities
    maccs_feats = [f for f in features_ordered if f.startswith('MACCS_')]
    cont_feats = [f for f in features_ordered if not f.startswith('MACCS_')]

    sim_scaler = StandardScaler()
    train_cont_sc = sim_scaler.fit_transform(train_df[cont_feats].values)
    ext_cont_sc = sim_scaler.transform(ext_df[cont_feats].values)
    
    nbrs = NearestNeighbors(n_neighbors=5, metric='euclidean').fit(train_cont_sc)
    dists, _ = nbrs.kneighbors(ext_cont_sc)
    normalized_dists = dists / np.sqrt(len(cont_feats))
    ext_df['sim_desc'] = np.mean(1 / (1 + normalized_dists), axis=1)

    train_maccs = train_df[maccs_feats].values.astype(bool)
    ext_maccs = ext_df[maccs_feats].values.astype(bool)
    tanimoto_sims = 1 - pairwise_distances(ext_maccs, train_maccs, metric='jaccard')
    ext_df['sim_fp'] = np.mean(np.sort(tanimoto_sims, axis=1)[:, -5:], axis=1)

    ext_df['sim_avg'] = (ext_df['sim_desc'] + ext_df['sim_fp']) / 2

    # Binning & Bin MAE
    bins = [0, 0.4, 0.6, 0.7, 0.8, 0.9, 1.01]
    labels = ["<40%", "40-60%", "60-70%", "70-80%", "80-90%", "90%+"]
    ext_df['Sim_Bin'] = pd.cut(ext_df['sim_avg'], bins=bins, labels=labels, right=False)
    
    # Calculate performance metrics
    global_metrics = calc_5_metrics(ext_df['LogS_Median'], ext_df['y_pred'])
    
    bin_stats = []
    for label in labels:
        subset = ext_df[ext_df['Sim_Bin'] == label]
        metrics = calc_5_metrics(subset['LogS_Median'], subset['y_pred'])
        bin_stats.append({
            'Similarity Bin': label,
            'N': len(subset),
            'R2': metrics['R2'],
            'RMSE': metrics['RMSE'],
            'MAE': metrics['MAE'],
            'Pct_07': metrics['Pct_07'],
            'Pct_10': metrics['Pct_10']
        })
    bin_df = pd.DataFrame(bin_stats)
    
    # Map Bin MAE back to individual drugs
    bin_mae_lookup = bin_df.set_index('Similarity Bin')['MAE'].to_dict()
    ext_df['Bin_MAE'] = ext_df['Sim_Bin'].map(bin_mae_lookup)

    print("\n--- STEP 6: Visualization & Reporting ---")
    # Color-coded scatter plot
    fig_scatter = plt.figure(figsize=(9, 7))
    scatter = sns.scatterplot(data=ext_df, x='LogS_Median', y='y_pred', hue='sim_avg', palette='viridis', s=80, edgecolor='black')
    
    # Add Colorbar
    norm = plt.Normalize(ext_df['sim_avg'].min(), ext_df['sim_avg'].max())
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    cbar = fig_scatter.colorbar(sm, ax=plt.gca())
    cbar.set_label('Average Similarity (0 to 1)', rotation=270, labelpad=15)
    
    scatter.legend_.remove() # Remove default legend since we have colorbar
    min_v, max_v = ext_df['LogS_Median'].min(), ext_df['LogS_Median'].max()
    plt.plot([min_v, max_v], [min_v, max_v], 'k--', lw=2)
    plt.xlabel("Actual LogS")
    plt.ylabel("Predicted LogS")
    b64_scatter = save_and_get_b64(fig_scatter, "external_scatter_sim.png")

    # Build Individual Drug Table rows
    ext_df = ext_df.sort_values('abs_err')
    drug_rows = ""
    for _, row in ext_df.iterrows():
        drug_rows += f"""<tr>
            <td><b>{row['Name']}</b></td>
            <td>{row['LogS_Median']:.2f}</td>
            <td>{row['y_pred']:.2f}</td>
            <td><b>{row['abs_err']:.3f}</b></td>
            <td>{row['sim_avg']*100:.1f}%</td>
            <td>{row['Sim_Bin']}</td>
            <td>{row['Bin_MAE']:.3f}</td>
        </tr>"""

    html = f"""
    <html><head><meta charset="UTF-8"><style>
        body {{ font-family: 'Segoe UI', sans-serif; background: #f4f7f9; color: #333; margin: 0; padding: 30px; }}
        h1, h2 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 5px; }}
        .section {{ background: white; padding: 25px; border-radius: 10px; box-shadow: 0 4px 10px rgba(0,0,0,0.05); margin-bottom: 30px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 0.95em; }}
        th, td {{ padding: 12px; border-bottom: 1px solid #eee; text-align: center; }}
        th {{ background: #2c3e50; color: white; }}
        td:first-child {{ text-align: left; }}
        .tracker-box {{ display: flex; justify-content: space-between; background: #ecf0f1; padding: 15px; border-radius: 8px; font-weight: bold; text-align: center; }}
        .tracker-item {{ flex: 1; }}
        .tracker-number {{ font-size: 1.5em; color: #2980b9; display: block; }}
        img {{ max-width: 100%; height: auto; display: block; margin: 0 auto; }}
    </style></head><body>

    <h1>External Validation Set Report</h1>
    
    <div class="section">
        <h2>1. Data Funnel & Tracking</h2>
        <div class="tracker-box">
            <div class="tracker-item">Original Structure Links<span class="tracker-number">{stats_tracker['original_structure_links']}</span></div>
            <div class="tracker-item">Passed Curation<span class="tracker-number">{stats_tracker['curated_drugs']}</span></div>
            <div class="tracker-item">Train/Test Excluded<span class="tracker-number">-{stats_tracker['overlap_total']}</span></div>
            <div class="tracker-item">Isolated External Set<span class="tracker-number">{stats_tracker['isolated_external']}</span></div>
            <div class="tracker-item">Matched in AqSolDB<span class="tracker-number" style="color:#2ecc71;">{stats_tracker['final_aqsol_matched']}</span></div>
        </div>
    </div>

    <div class="section">
        <h2>2. Global Performance on External Set</h2>
        <table>
            <tr><th>Subset</th><th>N</th><th>R²</th><th>RMSE</th><th>MAE</th><th>% Error ≤ 0.7</th><th>% Error ≤ 1.0</th></tr>
            <tr>
                <td><b>Total External Set</b></td>
                <td>{stats_tracker['final_aqsol_matched']}</td>
                <td>{global_metrics['R2']:.3f}</td>
                <td>{global_metrics['RMSE']:.3f}</td>
                <td>{global_metrics['MAE']:.3f}</td>
                <td>{global_metrics['Pct_07']:.1f}%</td>
                <td>{global_metrics['Pct_10']:.1f}%</td>
            </tr>
        </table>
    </div>

    <div class="section">
        <h2>3. Prediction Accuracy by Similarity Domain</h2>
        <img src="data:image/png;base64,{b64_scatter}" style="max-width: 800px; margin-bottom: 20px;">
        {bin_df.to_html(index=False, classes='table', float_format="%.3f", na_rep="-")}
    </div>

    <div class="section">
        <h2>4. Individual External Drug Errors</h2>
        <table>
            <tr><th>Drug Name</th><th>Actual LogS</th><th>Pred LogS</th><th>Absolute Error</th><th>Avg Sim</th><th>Sim Bin</th><th>Overall Bin MAE</th></tr>
            {drug_rows}
        </table>
    </div>

    </body></html>
    """
    with open(HTML_OUT, "w", encoding='utf-8') as f: f.write(html)
    
    # Final Console Printout
    print("\n" + "="*50)
    print("PIPELINE COMPLETE - DATA FUNNEL")
    print("="*50)
    print(f"1. Original Structure Links:  {stats_tracker['original_structure_links']}")
    print(f"2. Passed ML Curation:        {stats_tracker['curated_drugs']}")
    print(f"3. Excluded (In Train):      -{stats_tracker['overlap_train']}")
    print(f"4. Excluded (In Test):       -{stats_tracker['overlap_test']}")
    print(f"5. Isolated External Set:     {stats_tracker['isolated_external']}")
    print(f"6. Matched to AqSolDB LogS:   {stats_tracker['final_aqsol_matched']}")
    print("="*50)
    print(f"Report saved to: {HTML_OUT}")

if __name__ == "__main__":
    main()