import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import shap
from io import BytesIO
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# --- Configuration ---
TRAIN_FILE = 'train_data_final.csv'
TEST_FILE = 'test_data_final.csv'
MODEL_FILE = 'trained_model.joblib'
SCALER_FILE = 'scaler_final.joblib'
LINKS_FILE = 'path/to/DrugBankStructurelinks'
HTML_OUT = 'Drug_Explainability_Dashboard.html'
PLOTS_DIR = 'drugbank_plots'

if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)

def save_and_get_b64(fig, filename):
    filepath = os.path.join(PLOTS_DIR, filename)
    fig.savefig(filepath, format='png', bbox_inches='tight', dpi=120)
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=120)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def mol_to_svg(smi):
    try:
        m = Chem.MolFromSmiles(str(smi))
        if not m: return ""
        d = Draw.MolDraw2DSVG(250, 150)
        d.DrawMolecule(m)
        d.FinishDrawing()
        return d.GetDrawingText()
    except: return ""

def get_metrics(y_true, y_pred, name, n_count):
    if n_count < 2:
        return f"<tr><td>{name}</td><td>{n_count}</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>"
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    p07 = np.mean(np.abs(y_true - y_pred) <= 0.7) * 100
    p10 = np.mean(np.abs(y_true - y_pred) <= 1.0) * 100
    return f"<tr><td><b>{name}</b></td><td>{n_count}</td><td>{r2:.3f}</td><td>{rmse:.3f}</td><td>{mae:.3f}</td><td>{p07:.1f}%</td><td>{p10:.1f}%</td></tr>"

def main():
    print("Loading Data & Models...")
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    
    # FIXED: Extract features directly from columns
    features = [c for c in train_df.columns if c not in ['SMILES', 'LogS_Median']]

    # Define Feature Subsets for SHAP
    maccs_feats = [f for f in features if f.startswith('MACCS_')]
    polar_feats = ['OH_bonds', 'NH_bonds', 'SH_bonds', 'CO_bonds', 'CN_bonds', 'C_Halogen_bonds']
    ring_feats = ['ring_size_4', 'ring_size_5', 'ring_size_6', 'ring_size_7', 'ring_size_8', 'macrocycles']
    custom_labels = ['C_Aliphatic', 'C_Aromatic', 'O_Aromatic', 'single_bonds', 'double_bonds', 'triple_bonds', 'aromatic_bonds'] + polar_feats + ring_feats
    custom_feats = [f for f in features if len(f) <= 2 or f in custom_labels]

# Predict
    X_test_sc = scaler.transform(test_df[features].values)
    X_train_sc = scaler.transform(train_df[features].values)
    
    # Use .copy() to defragment the train dataframe before adding a column
    train_df = train_df.copy()
    train_df['y_pred'] = model.predict(X_train_sc)
    
    # Store all new test columns in a dictionary to prevent fragmentation
    new_cols = {}
    new_cols['y_pred'] = model.predict(X_test_sc)
    new_cols['abs_err'] = np.abs(test_df['LogS_Median'] - new_cols['y_pred'])
    new_cols['raw_err'] = new_cols['y_pred'] - test_df['LogS_Median']

    # --- Drug Matching ---
    print("Matching Approved Drugs...")
    links_df = pd.read_csv(LINKS_FILE)
    key_map = dict(zip(links_df['InChIKey'], links_df['Name']))
    
    def get_inchikey(smi):
        m = Chem.MolFromSmiles(smi)
        return Chem.MolToInchiKey(m) if m else None
        
    new_cols['InChIKey'] = test_df['SMILES'].apply(get_inchikey)
    new_cols['Drug_Name'] = new_cols['InChIKey'].map(key_map)
    new_cols['is_drug'] = new_cols['Drug_Name'].notna()

    # --- Lipinski Rule of 5 Calculations ---
    print("Calculating Lipinski Parameters...")
    new_cols['MW'] = test_df['SMILES'].apply(lambda x: Descriptors.MolWt(Chem.MolFromSmiles(x)))
    new_cols['LogP'] = test_df['SMILES'].apply(lambda x: Descriptors.MolLogP(Chem.MolFromSmiles(x)))
    new_cols['HBD'] = test_df['SMILES'].apply(lambda x: Descriptors.NumHDonors(Chem.MolFromSmiles(x)))
    new_cols['HBA'] = test_df['SMILES'].apply(lambda x: Descriptors.NumHAcceptors(Chem.MolFromSmiles(x)))
    new_cols['RotBonds'] = test_df['SMILES'].apply(lambda x: Descriptors.NumRotatableBonds(Chem.MolFromSmiles(x)))

    new_cols['Pass_MW'] = new_cols['MW'] <= 500
    new_cols['Pass_LogP'] = new_cols['LogP'] <= 5
    new_cols['Pass_HBD'] = new_cols['HBD'] <= 5
    new_cols['Pass_HBA'] = new_cols['HBA'] <= 10
    new_cols['Pass_RotBonds'] = new_cols['RotBonds'] <= 10
    
    new_cols['Lipinski_Violations'] = 5 - (new_cols['Pass_MW'].astype(int) + new_cols['Pass_LogP'].astype(int) + 
                                          new_cols['Pass_HBD'].astype(int) + new_cols['Pass_HBA'].astype(int) + new_cols['Pass_RotBonds'].astype(int))
    new_cols['Passes_All'] = new_cols['Lipinski_Violations'] == 0

    # Combine all new columns into test_df at exactly the same time
    new_cols_df = pd.DataFrame(new_cols, index=test_df.index)
    test_df = pd.concat([test_df, new_cols_df], axis=1)

    # --- Separate Performance Metrics Tables ---
    drug_df = test_df[test_df['is_drug']]
    range_df = test_df[(test_df['LogS_Median'] >= -5) & (test_df['LogS_Median'] <= -1)]
    drug_range_df = drug_df[(drug_df['LogS_Median'] >= -5) & (drug_df['LogS_Median'] <= -1)]

    metrics_global = ""
    metrics_global += get_metrics(train_df['LogS_Median'], train_df['y_pred'], "Overall Train Set", len(train_df))
    metrics_global += get_metrics(test_df['LogS_Median'], test_df['y_pred'], "Overall Test Set", len(test_df))
    metrics_global += get_metrics(range_df['LogS_Median'], range_df['y_pred'], "Test Set: LogS [-1 to -5]", len(range_df))

    metrics_drugs = ""
    metrics_drugs += get_metrics(drug_df['LogS_Median'], drug_df['y_pred'], "All Approved Drugs (Test)", len(drug_df))
    metrics_drugs += get_metrics(drug_range_df['LogS_Median'], drug_range_df['y_pred'], "Approved Drugs in LogS [-1 to -5]", len(drug_range_df))

    metrics_lip_global = ""
    metrics_lip_global += get_metrics(test_df[test_df['Pass_MW']]['LogS_Median'], test_df[test_df['Pass_MW']]['y_pred'], "Passes MW ≤ 500", len(test_df[test_df['Pass_MW']]))
    metrics_lip_global += get_metrics(test_df[test_df['Pass_LogP']]['LogS_Median'], test_df[test_df['Pass_LogP']]['y_pred'], "Passes LogP ≤ 5", len(test_df[test_df['Pass_LogP']]))
    metrics_lip_global += get_metrics(test_df[test_df['Pass_HBD']]['LogS_Median'], test_df[test_df['Pass_HBD']]['y_pred'], "Passes HBD ≤ 5", len(test_df[test_df['Pass_HBD']]))
    metrics_lip_global += get_metrics(test_df[test_df['Pass_HBA']]['LogS_Median'], test_df[test_df['Pass_HBA']]['y_pred'], "Passes HBA ≤ 10", len(test_df[test_df['Pass_HBA']]))
    metrics_lip_global += get_metrics(test_df[test_df['Pass_RotBonds']]['LogS_Median'], test_df[test_df['Pass_RotBonds']]['y_pred'], "Passes Rot Bonds ≤ 10", len(test_df[test_df['Pass_RotBonds']]))
    metrics_lip_global += get_metrics(test_df[test_df['Passes_All']]['LogS_Median'], test_df[test_df['Passes_All']]['y_pred'], "Passes ALL 5 Rules", len(test_df[test_df['Passes_All']]))

    metrics_lip_drugs = ""
    metrics_lip_drugs += get_metrics(drug_df[drug_df['Pass_MW']]['LogS_Median'], drug_df[drug_df['Pass_MW']]['y_pred'], "Drugs Passing MW ≤ 500", len(drug_df[drug_df['Pass_MW']]))
    metrics_lip_drugs += get_metrics(drug_df[drug_df['Pass_LogP']]['LogS_Median'], drug_df[drug_df['Pass_LogP']]['y_pred'], "Drugs Passing LogP ≤ 5", len(drug_df[drug_df['Pass_LogP']]))
    metrics_lip_drugs += get_metrics(drug_df[drug_df['Pass_HBD']]['LogS_Median'], drug_df[drug_df['Pass_HBD']]['y_pred'], "Drugs Passing HBD ≤ 5", len(drug_df[drug_df['Pass_HBD']]))
    metrics_lip_drugs += get_metrics(drug_df[drug_df['Pass_HBA']]['LogS_Median'], drug_df[drug_df['Pass_HBA']]['y_pred'], "Drugs Passing HBA ≤ 10", len(drug_df[drug_df['Pass_HBA']]))
    metrics_lip_drugs += get_metrics(drug_df[drug_df['Pass_RotBonds']]['LogS_Median'], drug_df[drug_df['Pass_RotBonds']]['y_pred'], "Drugs Passing Rot Bonds ≤ 10", len(drug_df[drug_df['Pass_RotBonds']]))
    metrics_lip_drugs += get_metrics(drug_df[drug_df['Passes_All']]['LogS_Median'], drug_df[drug_df['Passes_All']]['y_pred'], "Drugs Passing ALL 5 Rules", len(drug_df[drug_df['Passes_All']]))

    # --- GLOBAL VISUALIZATIONS (Blue and Orange) ---
    print("Generating Global Plots...")
    sns.set_theme(style="whitegrid")
    
    # 1. Global Scatter
    fig_scatter = plt.figure(figsize=(9, 7))
    sns.scatterplot(data=test_df[~test_df['is_drug']], x='LogS_Median', y='y_pred', color='#e67e22', alpha=0.5, label='Non-Drugs (Orange)', s=30)
    sns.scatterplot(data=test_df[test_df['is_drug']], x='LogS_Median', y='y_pred', color='#3498db', marker='D', s=80, edgecolor='black', label='Approved Drugs (Blue)')
    min_v, max_v = test_df['LogS_Median'].min(), test_df['LogS_Median'].max()
    plt.plot([min_v, max_v], [min_v, max_v], 'k--', lw=2)
    plt.xlabel("Actual LogS"); plt.ylabel("Predicted LogS")
    b64_scatter = save_and_get_b64(fig_scatter, "scatter_global.png")

    # 2. Window [-1 to -5] Scatter
    fig_zoom = plt.figure(figsize=(9, 7))
    sns.scatterplot(data=range_df[~range_df['is_drug']], x='LogS_Median', y='y_pred', color='#e67e22', alpha=0.6, s=40)
    sns.scatterplot(data=range_df[range_df['is_drug']], x='LogS_Median', y='y_pred', color='#3498db', marker='D', s=100, edgecolor='black')
    plt.plot([-5, -1], [-5, -1], 'k--', lw=2)
    plt.xlabel("Actual LogS"); plt.ylabel("Predicted LogS")
    b64_zoom = save_and_get_b64(fig_zoom, "scatter_zoomed.png")

    # 3. Window [-1 to -5] Error Distribution
    fig_dist = plt.figure(figsize=(9, 5))
    sns.kdeplot(data=range_df[~range_df['is_drug']], x='raw_err', fill=True, color='#e67e22', alpha=0.4, label='Non-Drugs (Orange)')
    sns.kdeplot(data=range_df[range_df['is_drug']], x='raw_err', fill=True, color='#3498db', alpha=0.6, label='Approved Drugs (Blue)')
    plt.axvline(0, color='black', linestyle='--')
    plt.xlabel("Error (Predicted - Actual)"); plt.legend()
    b64_dist = save_and_get_b64(fig_dist, "error_distribution.png")

    # 4. Lipinski Violations Violin Plot
    fig_viol = plt.figure(figsize=(9, 5))
    sns.violinplot(data=test_df, x='Lipinski_Violations', y='abs_err', color='#3498db', inner='quartile')
    plt.xlabel("Number of Lipinski Violations"); plt.ylabel("Absolute Prediction Error")
    b64_viol = save_and_get_b64(fig_viol, "lipinski_violin.png")

    # --- GLOBAL SHAP PLOTS ---
    print("Generating SHAP Summaries...")
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_test_sc)
    
    # 1. Overall
    fig_shap_all = plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_vals, X_test_sc, feature_names=features, show=False)
    b64_shap_all = save_and_get_b64(fig_shap_all, "shap_summary_all.png")
    
    # 2. Custom features
    custom_idx = [features.index(f) for f in custom_feats]
    fig_shap_cust = plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_vals[:, custom_idx], X_test_sc[:, custom_idx], feature_names=custom_feats, show=False)
    b64_shap_cust = save_and_get_b64(fig_shap_cust, "shap_summary_custom.png")
    
    # 3. MACCS Keys
    maccs_idx = [features.index(f) for f in maccs_feats]
    fig_shap_maccs = plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_vals[:, maccs_idx], X_test_sc[:, maccs_idx], feature_names=maccs_feats, max_display=15, show=False)
    b64_shap_maccs = save_and_get_b64(fig_shap_maccs, "shap_summary_maccs.png")

    # 4. Window [-1 to -5] Only
    range_mask = ((test_df['LogS_Median'] >= -5) & (test_df['LogS_Median'] <= -1)).values
    fig_shap_window = plt.figure(figsize=(10, 8))
    if range_mask.sum() > 0:
        shap.summary_plot(shap_vals[range_mask], X_test_sc[range_mask], feature_names=features, show=False)
    b64_shap_window = save_and_get_b64(fig_shap_window, "shap_summary_window.png")

    # 5. Approved Drugs Only
    drug_mask = test_df['is_drug'].values
    fig_shap_drug = plt.figure(figsize=(10, 8))
    if drug_mask.sum() > 0:
        shap.summary_plot(shap_vals[drug_mask], X_test_sc[drug_mask], feature_names=features, show=False)
    b64_shap_drug = save_and_get_b64(fig_shap_drug, "shap_summary_drugs.png")


    # --- LOCAL SHAP WATERFALLS (7 Best, 7 Mid, 7 Worst) ---
    print("Generating Individual SHAP Waterfalls for Drugs...")
    drug_df = drug_df.sort_values('abs_err').reset_index(drop=True)
    
    best_idx = list(range(0, min(7, len(drug_df))))
    worst_idx = list(range(max(0, len(drug_df)-7), len(drug_df)))
    mid_start = max(0, (len(drug_df)//2) - 3)
    mid_idx = list(range(mid_start, min(len(drug_df), mid_start+7)))
    
    selected_indices = sorted(list(set(best_idx + mid_idx + worst_idx)))
    expected_val = explainer.expected_value[0] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value
    
    cards_html = ""
    for idx in selected_indices:
        row = drug_df.iloc[idx]
        group = "TOP 7 BEST" if idx in best_idx else ("7 WORST" if idx in worst_idx else "MEDIAN 7")
        orig_idx = test_df.index[test_df['SMILES'] == row['SMILES']].tolist()[0]
        
        exp = shap.Explanation(values=shap_vals[orig_idx], base_values=expected_val, data=X_test_sc[orig_idx], feature_names=features)
        fig_waterfall = plt.figure(figsize=(6, 5))
        shap.plots.waterfall(exp, max_display=8, show=False)
        
        # Modify the text objects in the waterfall plot
        ax = plt.gca()
        for text in ax.texts:
            col = text.get_color()
            # SHAP uses '#999999' for the feature values. Convert to black.
            if col == '#999999' or col == 'grey' or col == 'gray':
                text.set_color('black')
            # Embolden existing black text (or text we just turned black)
            text.set_fontweight('bold')
            
        # Bold the axis tick labels for extra readability
        for tick in ax.get_yticklabels():
            tick.set_color('black')
            tick.set_fontweight('bold')
        for tick in ax.get_xticklabels():
            tick.set_color('black')
            tick.set_fontweight('bold')

        plt.tight_layout()
        
        safe_name = "".join([c for c in str(row['Drug_Name']) if c.isalnum() or c in " -_"]).strip()
        b64_wf = save_and_get_b64(fig_waterfall, f"waterfall_{safe_name}_{idx}.png")
        
        svg = mol_to_svg(row['SMILES'])
        
        cards_html += f"""
        <div class="card">
            <div class="card-header">{group}: {row['Drug_Name']}</div>
            <div class="card-img">{svg}</div>
            <div class="card-metrics">
                <b>Actual:</b> {row['LogS_Median']:.2f} | <b>Pred:</b> {row['y_pred']:.2f}<br>
                <b>Error:</b> <span style="color:{'#2ecc71' if row['abs_err'] <= 0.7 else '#e74c3c'}">{row['abs_err']:.2f}</span>
            </div>
            <img src="data:image/png;base64,{b64_wf}" style="width:100%; border-top: 1px solid #ddd; margin-top:10px;">
        </div>
        """

    # --- HTML ASSEMBLY ---
    print("Saving Dashboard...")
    table_header = "<tr><th>Subset</th><th>N (Count)</th><th>R²</th><th>RMSE</th><th>MAE</th><th>% Error ≤ 0.7</th><th>% Error ≤ 1.0</th></tr>"
    
    html = f"""
    <html><head><meta charset="UTF-8"><style>
        body {{ font-family: 'Segoe UI', sans-serif; background: #f4f7f9; color: #333; margin: 0; padding: 30px; }}
        h1, h2 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 5px; }}
        h3 {{ color: #2980b9; margin-top: 25px; }}
        .section {{ background: white; padding: 25px; border-radius: 10px; box-shadow: 0 4px 10px rgba(0,0,0,0.05); margin-bottom: 30px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; margin-bottom: 10px; font-size: 0.95em; }}
        th, td {{ padding: 12px; border-bottom: 1px solid #eee; text-align: center; }}
        th {{ background: #2c3e50; color: white; }}
        td:first-child {{ text-align: left; width: 35%; }}
        .flex-container {{ display: flex; flex-wrap: wrap; gap: 20px; justify-content: center; }}
        .flex-item {{ flex: 1 1 45%; min-width: 400px; text-align: center; }}
        .flex-item img {{ max-width: 100%; border-radius: 8px; border: 1px solid #ddd; }}
        
        .grid-container {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(350px, 1fr)); gap: 25px; }}
        .card {{ background: white; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); overflow: hidden; display: flex; flex-direction: column; }}
        .card-header {{ background: #3498db; color: white; padding: 12px; font-weight: bold; font-size: 1.1em; text-align: center; }}
        .card-img {{ background: #fff; padding: 15px; text-align: center; border-bottom: 1px solid #eee; height: 160px; display:flex; align-items:center; justify-content:center; }}
        .card-metrics {{ padding: 15px; text-align: center; font-size: 1.1em; background: #fcfcfc; }}
    </style></head><body>

    <h1>Mega Model: Drug Explainability Dashboard</h1>

    <div class="section">
        <h2>1. Global & Lipinski Performance Metrics</h2>
        
        <h3>Global Overview</h3>
        <table>
            {table_header}
            {metrics_global}
        </table>
        
        <h3>Approved Drugs Accuracy</h3>
        <table>
            {table_header}
            {metrics_drugs}
        </table>

        <h3>Lipinski "Rule of 5" Focus (Entire Test Set)</h3>
        <table>
            {table_header}
            {metrics_lip_global}
        </table>

        <h3>Lipinski "Rule of 5" Focus (Approved Drugs Only)</h3>
        <table>
            {table_header}
            {metrics_lip_drugs}
        </table>
    </div>

    <div class="section">
        <h2>2. Global Error Visualizations (Blue = Drugs, Orange = Non-Drugs)</h2>
        <div class="flex-container">
            <div class="flex-item"><img src="data:image/png;base64,{b64_scatter}"></div>
            <div class="flex-item"><img src="data:image/png;base64,{b64_viol}"></div>
        </div>
        <h3 style="text-align:center; margin-top:30px;">Focus: The [-1 to -5] "Drug-Like" LogS Window</h3>
        <div class="flex-container">
            <div class="flex-item"><img src="data:image/png;base64,{b64_zoom}"></div>
            <div class="flex-item"><img src="data:image/png;base64,{b64_dist}"></div>
        </div>
    </div>

    <div class="section">
        <h2>3. Global Feature Importance (SHAP)</h2>
        <div class="flex-container">
            <div class="flex-item">
                <h3>Overall Top 20 Features</h3>
                <img src="data:image/png;base64,{b64_shap_all}">
            </div>
            <div class="flex-item">
                <h3>Custom Generative "Knobs"</h3>
                <img src="data:image/png;base64,{b64_shap_cust}">
            </div>
        </div>
        <div class="flex-container" style="margin-top: 30px;">
            <div class="flex-item">
                <h3>Features Driving the [-1 to -5] LogS Window</h3>
                <img src="data:image/png;base64,{b64_shap_window}">
            </div>
            <div class="flex-item">
                <h3>Features Driving Approved Drugs</h3>
                <img src="data:image/png;base64,{b64_shap_drug}">
            </div>
        </div>
        <div class="flex-container" style="margin-top: 30px;">
            <div class="flex-item" style="flex: 1 1 100%;">
                <h3>MACCS Keys Logic</h3>
                <img src="data:image/png;base64,{b64_shap_maccs}" style="max-width: 800px;">
            </div>
        </div>
    </div>

    <div class="section" style="background: transparent; box-shadow: none; padding: 0;">
        <h2>4. Local Explainability: Individual Drug Predictions</h2>
        <p>Waterfall plots for the 7 Best, 7 Median, and 7 Worst predicted approved drugs. These plots display exactly how much each feature shifted the prediction up (red) or down (blue) from the baseline expected solubility.</p>
        <div class="grid-container">
            {cards_html}
        </div>
    </div>

    </body></html>
    """
    with open(HTML_OUT, "w", encoding='utf-8') as f: f.write(html)
    print(f"Done! Dashboard saved to {HTML_OUT}")

if __name__ == "__main__":
    main()